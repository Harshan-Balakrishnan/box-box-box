import argparse
import copy
import itertools
import json
from pathlib import Path

import solution.race_simulator as race_simulator
from tools.evaluate_historical import build_pattern, evaluate_races, load_historical_races


BASE_OVERRIDES = {
    "SOFT": {
        "compound_delta": -0.96,
        "fresh_bonus": -0.31,
        "second_lap_bonus": 0.0,
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
        "second_lap_bonus": 0.0,
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

TARGET_PATTERNS = {
    "hard_to_final5": {
        *(f"HARD:{length} -> SOFT:5" for length in range(30, 41)),
        *(f"HARD:{length} -> MEDIUM:5" for length in range(30, 41)),
    },
    "soft7_9_to_long_hard": {
        *(f"SOFT:{soft_len} -> HARD:{hard_len}" for soft_len in (7, 8, 9) for hard_len in range(28, 37)),
    },
}


def deep_copy_params(params):
    return {compound: values.copy() for compound, values in params.items()}


def build_base_params():
    params = deep_copy_params(race_simulator.COMPOUND_PARAMS)
    for compound, updates in BASE_OVERRIDES.items():
        params[compound].update(updates)
    return params


def total_rank_error(summary):
    return sum(row["total_abs_rank_error"] for row in summary["worst_races"])


def build_hard_shape_lap_time(candidate):
    base_lap_time_fn = race_simulator.lap_time
    normalize_compound = race_simulator.normalize_compound

    def lap_time_with_hard_shape(base_lap_time, compound, tire_age, track_temp):
        total = base_lap_time_fn(base_lap_time, compound, tire_age, track_temp)

        if normalize_compound(compound) != "HARD":
            return total

        shape_adjustment = 0.0

        if tire_age >= candidate["mid_start"]:
            mid_laps = min(tire_age, candidate["mid_end"]) - candidate["mid_start"] + 1
            if mid_laps > 0:
                shape_adjustment += candidate["mid_penalty_per_lap"] * mid_laps

        if tire_age >= candidate["relief_start"]:
            relief_laps = tire_age - candidate["relief_start"] + 1
            if relief_laps > 0:
                shape_adjustment += candidate["relief_per_lap"] * relief_laps

        return total + shape_adjustment

    return lap_time_with_hard_shape


def evaluate_with_candidate(races, base_params, candidate):
    original_params = race_simulator.COMPOUND_PARAMS
    original_lap_time = race_simulator.lap_time
    try:
        race_simulator.COMPOUND_PARAMS = deep_copy_params(base_params)
        race_simulator.lap_time = build_hard_shape_lap_time(candidate)
        summary = evaluate_races(races)
    finally:
        race_simulator.COMPOUND_PARAMS = original_params
        race_simulator.lap_time = original_lap_time
    return summary


def collect_target_metrics(races):
    predicted_by_race = {}
    for race in races:
        predicted = race_simulator.predict_finishing_positions(copy.deepcopy(race))
        predicted_by_race[race["race_id"]] = predicted["finishing_positions"]

    metrics = {}
    for group_name in TARGET_PATTERNS:
        metrics[group_name] = {
            "samples": 0,
            "avg_signed_error": 0.0,
            "avg_abs_error": 0.0,
        }

    for race in races:
        predicted_order = predicted_by_race[race["race_id"]]
        actual_order = race["finishing_positions"]
        predicted_pos = {driver_id: index + 1 for index, driver_id in enumerate(predicted_order)}
        actual_pos = {driver_id: index + 1 for index, driver_id in enumerate(actual_order)}
        total_laps = int(race["race_config"]["total_laps"])

        for strategy in race["strategies"].values():
            pattern = build_pattern(strategy, total_laps)
            driver_id = strategy["driver_id"]
            signed_error = predicted_pos[driver_id] - actual_pos[driver_id]
            abs_error = abs(signed_error)

            for group_name, patterns in TARGET_PATTERNS.items():
                if pattern in patterns:
                    metrics[group_name]["samples"] += 1
                    metrics[group_name]["avg_signed_error"] += signed_error
                    metrics[group_name]["avg_abs_error"] += abs_error

    for group_name, row in metrics.items():
        samples = row["samples"]
        if samples:
            row["avg_signed_error"] /= samples
            row["avg_abs_error"] /= samples

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Search a two-kink HARD stint shape using historical races only."
    )
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
        baseline_target_metrics = collect_target_metrics(races)
    finally:
        race_simulator.COMPOUND_PARAMS = original_params

    print("Baseline result")
    print("===============")
    print(f"Races evaluated:      {baseline_summary['total_races']}")
    print(f"Exact matches:        {baseline_summary['exact_matches']}")
    print(f"Exact-match accuracy: {baseline_summary['exact_match_accuracy'] * 100:.2f}%")
    print(f"Total rank error:     {total_rank_error(baseline_summary)}")
    print("Target pattern metrics:")
    print(json.dumps(baseline_target_metrics, indent=2))
    print()

    search_space = {
        "mid_start": [20, 21, 22, 23, 24],
        "mid_end": [25, 26, 27, 28, 29],
        "mid_penalty_per_lap": [0.00, 0.01, 0.02, 0.03],
        "relief_start": [28, 29, 30, 31, 32],
        "relief_per_lap": [0.00, -0.01, -0.02, -0.03, -0.04],
    }

    keys = list(search_space.keys())
    total_candidates = 1
    for key in keys:
        total_candidates *= len(search_space[key])

    results = []
    checked = 0
    for values in itertools.product(*(search_space[key] for key in keys)):
        candidate = dict(zip(keys, values))

        if candidate["mid_end"] < candidate["mid_start"]:
            continue
        if candidate["relief_start"] <= candidate["mid_start"]:
            continue
        if candidate["relief_start"] <= candidate["mid_end"] and candidate["relief_per_lap"] == 0.0:
            continue
        if candidate["mid_penalty_per_lap"] == 0.0 and candidate["relief_per_lap"] == 0.0:
            continue

        checked += 1
        if checked % 50 == 0:
            print(f"checked {checked}/{total_candidates}")

        summary = evaluate_with_candidate(races, base_params, candidate)

        original_params = race_simulator.COMPOUND_PARAMS
        original_lap_time = race_simulator.lap_time
        try:
            race_simulator.COMPOUND_PARAMS = deep_copy_params(base_params)
            race_simulator.lap_time = build_hard_shape_lap_time(candidate)
            target_metrics = collect_target_metrics(races)
        finally:
            race_simulator.COMPOUND_PARAMS = original_params
            race_simulator.lap_time = original_lap_time

        results.append({
            "candidate": candidate,
            "exact_matches": summary["exact_matches"],
            "exact_match_accuracy": summary["exact_match_accuracy"],
            "total_rank_error": total_rank_error(summary),
            "target_metrics": target_metrics,
        })

    results.sort(
        key=lambda row: (
            -row["exact_matches"],
            row["total_rank_error"],
            abs(row["target_metrics"]["hard_to_final5"]["avg_signed_error"]),
            abs(row["target_metrics"]["soft7_9_to_long_hard"]["avg_signed_error"]),
        )
    )

    print("Top candidates")
    print("==============")
    for index, row in enumerate(results[: args.show_top], start=1):
        print(
            f"{index}. exact_matches={row['exact_matches']}, "
            f"accuracy={row['exact_match_accuracy'] * 100:.2f}%, "
            f"total_rank_error={row['total_rank_error']}"
        )
        print(json.dumps(row["candidate"], indent=2))
        print(json.dumps(row["target_metrics"], indent=2))
        print()

    best = results[0]
    print("Best HARD ultra-long shape candidate")
    print("====================================")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()