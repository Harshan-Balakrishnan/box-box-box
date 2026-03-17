import argparse
import json
from pathlib import Path

import solution.race_simulator as race_simulator
from tools.evaluate_historical import load_historical_races, evaluate_races


BASE_OVERRIDES = {
    "SOFT": {
        "compound_delta": -0.96,
        "fresh_bonus": -0.35,
        "second_lap_bonus": -0.04,
        "grace_laps": 1,
        "linear_deg": 0.03,
        "quadratic_deg": 0.0022,
        "temp_deg": 0.0024,
        "sprint_window_start": 3,
        "sprint_window_end": 5,
        "sprint_window_bonus": -0.5,
        "fade_start": 7,
        "fade_per_lap": 0.2,
    },
    "MEDIUM": {
        "compound_delta": 0.0,
        "fresh_bonus": -0.03,
        "second_lap_bonus": -0.03,
        "grace_laps": 2,
        "linear_deg": 0.016,
        "quadratic_deg": 0.001,
        "temp_deg": 0.0014,
        "late_stint_start": 999,
        "late_stint_deg": 0.0,
    },
    "HARD": {
        "compound_delta": 0.68,
        "fresh_bonus": -0.08,
        "second_lap_bonus": -0.02,
        "grace_laps": 3,
        "linear_deg": 0.01,
        "quadratic_deg": 0.0005,
        "temp_deg": 0.0008,
        "late_stint_start": 13,
        "late_stint_deg": 0.0003,
    },
}


def deep_copy_params(params):
    return {compound: values.copy() for compound, values in params.items()}


def build_base_params():
    params = deep_copy_params(race_simulator.COMPOUND_PARAMS)
    for compound, updates in BASE_OVERRIDES.items():
        params[compound].update(updates)
    return params


def build_stints(strategy, total_laps):
    stints = []
    current_tire = str(strategy["starting_tire"]).upper()
    last_lap = 0

    for stop in strategy.get("pit_stops", []):
        stop_lap = int(stop["lap"])
        stints.append({
            "tire": current_tire,
            "length": stop_lap - last_lap,
        })
        current_tire = str(stop["to_tire"]).upper()
        last_lap = stop_lap

    stints.append({
        "tire": current_tire,
        "length": total_laps - last_lap,
    })
    return stints


def make_predictor(candidate):
    base_lap_time_fn = race_simulator.lap_time
    normalize_compound = race_simulator.normalize_compound

    def simulate_driver_with_bonus(race_config, strategy):
        total_laps = int(race_config["total_laps"])
        base_lap_time = float(race_config["base_lap_time"])
        pit_lane_time = float(race_config["pit_lane_time"])
        track_temp = float(race_config["track_temp"])

        stints = build_stints(strategy, total_laps)

        current_tire = normalize_compound(strategy["starting_tire"])
        tire_age = 0
        total_time = 0.0
        stint_index = 0

        pit_map = {}
        for stop in strategy.get("pit_stops", []):
            pit_map[int(stop["lap"])] = stop

        final_bonus = 0.0
        if len(stints) >= 2:
            prev_stint = stints[-2]
            final_stint = stints[-1]
            final_tire = normalize_compound(final_stint["tire"])

            if (
                prev_stint["tire"] == "HARD"
                and candidate["hard_min"] <= prev_stint["length"] <= candidate["hard_max"]
                and final_stint["length"] == 5
            ):
                if final_tire == "SOFT":
                    final_bonus = candidate["soft_bonus"]
                elif final_tire == "MEDIUM":
                    final_bonus = candidate["medium_bonus"]

        for lap in range(1, total_laps + 1):
            tire_age += 1
            lap_bonus = final_bonus if stint_index == len(stints) - 1 else 0.0

            total_time += base_lap_time_fn(base_lap_time, current_tire, tire_age, track_temp) + lap_bonus

            if lap in pit_map:
                stop = pit_map[lap]
                total_time += pit_lane_time
                current_tire = normalize_compound(stop["to_tire"])
                tire_age = 0
                stint_index += 1

        return total_time

    def predict_with_bonus(data):
        race_config = data["race_config"]
        results = []

        for strategy in data["strategies"].values():
            driver_id = strategy["driver_id"]
            total_time = simulate_driver_with_bonus(race_config, strategy)
            results.append((total_time, driver_id))

        results.sort(key=lambda item: (item[0], item[1]))
        return {
            "race_id": data["race_id"],
            "finishing_positions": [driver_id for _, driver_id in results],
        }

    return predict_with_bonus


def evaluate_with_candidate(races, base_params, candidate):
    original_params = race_simulator.COMPOUND_PARAMS
    original_predict = race_simulator.predict_finishing_positions
    try:
        race_simulator.COMPOUND_PARAMS = deep_copy_params(base_params)
        race_simulator.predict_finishing_positions = make_predictor(candidate)
        summary = evaluate_races(races)
    finally:
        race_simulator.COMPOUND_PARAMS = original_params
        race_simulator.predict_finishing_positions = original_predict
    return summary


def total_rank_error(summary):
    return sum(row["total_abs_rank_error"] for row in summary["worst_races"])


def main():
    parser = argparse.ArgumentParser(description="Search final-5 bonus after long HARD stint.")
    parser.add_argument("--historical-dir", default="data/historical_races")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--show-top", type=int, default=10)
    args = parser.parse_args()

    races = load_historical_races(Path(args.historical_dir), args.limit)
    base_params = build_base_params()

    original_params = race_simulator.COMPOUND_PARAMS
    try:
        race_simulator.COMPOUND_PARAMS = deep_copy_params(base_params)
        baseline_summary = evaluate_races(races)
    finally:
        race_simulator.COMPOUND_PARAMS = original_params

    print("Baseline result")
    print("===============")
    print(f"Races evaluated:      {baseline_summary['total_races']}")
    print(f"Exact matches:        {baseline_summary['exact_matches']}")
    print(f"Exact-match accuracy: {baseline_summary['exact_match_accuracy'] * 100:.2f}%")
    print(f"Total rank error:     {total_rank_error(baseline_summary)}")
    print()

    hard_min_values = [31, 32, 33]
    hard_max_values = [36, 38, 40]
    soft_bonus_values = [0.0, -0.01, -0.02, -0.03]
    medium_bonus_values = [0.0, -0.01, -0.02, -0.03]

    results = []
    for hard_min in hard_min_values:
        for hard_max in hard_max_values:
            if hard_min > hard_max:
                continue
            for soft_bonus in soft_bonus_values:
                for medium_bonus in medium_bonus_values:
                    candidate = {
                        "hard_min": hard_min,
                        "hard_max": hard_max,
                        "soft_bonus": soft_bonus,
                        "medium_bonus": medium_bonus,
                    }
                    summary = evaluate_with_candidate(races, base_params, candidate)
                    results.append({
                        "candidate": candidate,
                        "exact_matches": summary["exact_matches"],
                        "exact_match_accuracy": summary["exact_match_accuracy"],
                        "total_rank_error": total_rank_error(summary),
                    })

    results.sort(key=lambda row: (-row["exact_matches"], row["total_rank_error"]))

    print("Top candidates")
    print("==============")
    for i, row in enumerate(results[:args.show_top], start=1):
        print(
            f"{i}. exact_matches={row['exact_matches']}, "
            f"accuracy={row['exact_match_accuracy'] * 100:.2f}%, "
            f"total_rank_error={row['total_rank_error']}"
        )
        print(json.dumps(row["candidate"], indent=2))
        print()


if __name__ == "__main__":
    main()