import argparse
import itertools
import json
from pathlib import Path

import solution.race_simulator as race_simulator
from tools.evaluate_historical import load_historical_races, evaluate_races


BASE_OVERRIDES = {
    "SOFT": {
        "compound_delta": -0.98,
        "fresh_bonus": -0.35,
        "second_lap_bonus": -0.12,
        "grace_laps": 1,
        "linear_deg": 0.028,
        "quadratic_deg": 0.0024,
        "temp_deg": 0.0024,
        "sprint_window_start": 3,
        "sprint_window_end": 5,
        "sprint_window_bonus": -0.5,
        "fade_start": 5,
        "fade_per_lap": 0.25,
    },
    "MEDIUM": {
        "compound_delta": 0.0,
        "fresh_bonus": -0.18,
        "second_lap_bonus": -0.05,
        "grace_laps": 2,
        "linear_deg": 0.018,
        "quadratic_deg": 0.0012,
        "temp_deg": 0.0014,
        "late_stint_start": 999,
        "late_stint_deg": 0.0,
    },
    "HARD": {
        "compound_delta": 0.72,
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
    params["SOFT"]["compound_delta"] = candidate["soft_compound_delta"]
    params["SOFT"]["linear_deg"] = candidate["soft_linear_deg"]
    params["SOFT"]["quadratic_deg"] = candidate["soft_quadratic_deg"]
    params["SOFT"]["fade_start"] = candidate["soft_fade_start"]
    params["SOFT"]["fade_per_lap"] = candidate["soft_fade_per_lap"]
    return params


def evaluate_with_params(races, params):
    original_params = race_simulator.COMPOUND_PARAMS
    try:
        race_simulator.COMPOUND_PARAMS = params
        summary = evaluate_races(races)
    finally:
        race_simulator.COMPOUND_PARAMS = original_params
    return summary


def main():
    parser = argparse.ArgumentParser(description="Search core SOFT parameters.")
    parser.add_argument("--historical-dir", default="data/historical_races")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--show-top", type=int, default=10)
    args = parser.parse_args()

    races = load_historical_races(Path(args.historical_dir), args.limit)
    base_params = build_base_params()

    search_space = {
        "soft_compound_delta": [-1.08, -1.05, -1.03, -1.01, -0.98],
        "soft_linear_deg": [0.026, 0.028, 0.030, 0.032, 0.034],
        "soft_quadratic_deg": [0.0016, 0.0018, 0.0020, 0.0022, 0.0024],
        "soft_fade_start": [5, 6, 7, 8],
        "soft_fade_per_lap": [0.25, 0.30, 0.35, 0.40, 0.45],
    }

    baseline_summary = evaluate_with_params(races, base_params)
    baseline_rank_error = sum(row["total_abs_rank_error"] for row in baseline_summary["worst_races"])

    print("Baseline result")
    print("===============")
    print(f"Races evaluated:      {baseline_summary['total_races']}")
    print(f"Exact matches:        {baseline_summary['exact_matches']}")
    print(f"Exact-match accuracy: {baseline_summary['exact_match_accuracy'] * 100:.2f}%")
    print(f"Total rank error:     {baseline_rank_error}")
    print()

    results = []
    keys = list(search_space.keys())

    for values in itertools.product(*(search_space[key] for key in keys)):
        candidate = dict(zip(keys, values))
        params = apply_candidate(base_params, candidate)
        summary = evaluate_with_params(races, params)
        total_rank_error = sum(row["total_abs_rank_error"] for row in summary["worst_races"])

        results.append({
            "candidate": candidate,
            "exact_matches": summary["exact_matches"],
            "exact_match_accuracy": summary["exact_match_accuracy"],
            "total_rank_error": total_rank_error,
        })

    results.sort(
        key=lambda row: (
            -row["exact_matches"],
            row["total_rank_error"],
        )
    )

    print("Top candidates")
    print("==============")
    for index, row in enumerate(results[:args.show_top], start=1):
        print(f"{index}. exact_matches={row['exact_matches']}, "
              f"accuracy={row['exact_match_accuracy'] * 100:.2f}%, "
              f"total_rank_error={row['total_rank_error']}")
        print(json.dumps(row["candidate"], indent=2))
        print()

    best = results[0]
    best_params = apply_candidate(base_params, best["candidate"])

    print("Best SOFT parameter set to try next")
    print("==================================")
    print(json.dumps(best_params["SOFT"], indent=2))


if __name__ == "__main__":
    main()