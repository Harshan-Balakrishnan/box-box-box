import argparse
import itertools
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


def apply_candidate(base_params, candidate):
    params = deep_copy_params(base_params)
    params["SOFT"]["fresh_bonus"] = candidate["soft_fresh_bonus"]
    params["SOFT"]["second_lap_bonus"] = candidate["soft_second_lap_bonus"]
    params["MEDIUM"]["fresh_bonus"] = candidate["medium_fresh_bonus"]
    params["MEDIUM"]["second_lap_bonus"] = candidate["medium_second_lap_bonus"]
    return params


def evaluate_with_params(races, params):
    original_params = race_simulator.COMPOUND_PARAMS
    try:
        race_simulator.COMPOUND_PARAMS = params
        summary = evaluate_races(races)
    finally:
        race_simulator.COMPOUND_PARAMS = original_params
    return summary


def total_rank_error(summary):
    return sum(row["total_abs_rank_error"] for row in summary["worst_races"])


def main():
    parser = argparse.ArgumentParser(description="Small local search around freshness bonuses.")
    parser.add_argument("--historical-dir", default="data/historical_races")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--show-top", type=int, default=10)
    args = parser.parse_args()

    races = load_historical_races(Path(args.historical_dir), args.limit)
    base_params = build_base_params()

    search_space = {
        "soft_fresh_bonus": [-0.39, -0.37, -0.35, -0.33, -0.31],
        "soft_second_lap_bonus": [-0.08, -0.06, -0.04, -0.02, 0.0],
        "medium_fresh_bonus": [-0.07, -0.05, -0.03, -0.01, 0.0],
        "medium_second_lap_bonus": [-0.07, -0.05, -0.03, -0.01, 0.0],
    }

    baseline_summary = evaluate_with_params(races, base_params)

    print("Baseline result")
    print("===============")
    print(f"Races evaluated:      {baseline_summary['total_races']}")
    print(f"Exact matches:        {baseline_summary['exact_matches']}")
    print(f"Exact-match accuracy: {baseline_summary['exact_match_accuracy'] * 100:.2f}%")
    print(f"Total rank error:     {total_rank_error(baseline_summary)}")
    print()

    results = []
    keys = list(search_space.keys())
    total = 1
    for key in keys:
        total *= len(search_space[key])

    count = 0
    for values in itertools.product(*(search_space[key] for key in keys)):
        count += 1
        if count % 20 == 0:
            print(f"checked {count}/{total}")

        candidate = dict(zip(keys, values))
        params = apply_candidate(base_params, candidate)
        summary = evaluate_with_params(races, params)

        results.append({
            "candidate": candidate,
            "exact_matches": summary["exact_matches"],
            "exact_match_accuracy": summary["exact_match_accuracy"],
            "total_rank_error": total_rank_error(summary),
        })

    results.sort(key=lambda row: (-row["exact_matches"], row["total_rank_error"]))

    print("Top candidates")
    print("==============")
    for index, row in enumerate(results[:args.show_top], start=1):
        print(
            f"{index}. exact_matches={row['exact_matches']}, "
            f"accuracy={row['exact_match_accuracy'] * 100:.2f}%, "
            f"total_rank_error={row['total_rank_error']}"
        )
        print(json.dumps(row["candidate"], indent=2))
        print()

    best = results[0]
    best_params = apply_candidate(base_params, best["candidate"])

    print("Best freshness candidate")
    print("========================")
    print(json.dumps({
        "SOFT": {
            "fresh_bonus": best_params["SOFT"]["fresh_bonus"],
            "second_lap_bonus": best_params["SOFT"]["second_lap_bonus"],
        },
        "MEDIUM": {
            "fresh_bonus": best_params["MEDIUM"]["fresh_bonus"],
            "second_lap_bonus": best_params["MEDIUM"]["second_lap_bonus"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()