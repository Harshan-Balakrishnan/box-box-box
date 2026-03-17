import argparse
import copy
import itertools
import json
from pathlib import Path

import solution.race_simulator as race_simulator
from tools.evaluate_historical import load_historical_races, evaluate_races


BASE_OVERRIDES = {
    "SOFT": {
        "compound_delta": -1.03,
        "linear_deg": 0.03,
        "quadratic_deg": 0.002,
        "sprint_window_bonus": -0.5,
        "fade_start": 6,
        "fade_per_lap": 0.4,
    },
    "MEDIUM": {
        "linear_deg": 0.018,
        "quadratic_deg": 0.0012,
    },
    "HARD": {
        "linear_deg": 0.012,
        "quadratic_deg": 0.0006,
        "late_stint_start": 12,
        "late_stint_deg": 0.0005,
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
    params["HARD"]["linear_deg"] = candidate["hard_linear_deg"]
    params["HARD"]["quadratic_deg"] = candidate["hard_quadratic_deg"]
    params["HARD"]["late_stint_start"] = candidate["hard_late_stint_start"]
    params["HARD"]["late_stint_deg"] = candidate["hard_late_stint_deg"]
    return params


def score_summary(summary):
    total_rank_error = sum(row["total_abs_rank_error"] for row in summary["worst_races"])
    exact_matches = summary["exact_matches"]
    return exact_matches, -total_rank_error


def evaluate_with_params(races, params):
    original_params = race_simulator.COMPOUND_PARAMS
    try:
        race_simulator.COMPOUND_PARAMS = params
        summary = evaluate_races(races)
    finally:
        race_simulator.COMPOUND_PARAMS = original_params
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Search a small set of core HARD parameters using historical races only."
    )
    parser.add_argument(
        "--historical-dir",
        default="data/historical_races",
        help="Folder containing historical race JSON files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="How many historical races to evaluate.",
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=10,
        help="How many best candidates to print at the end.",
    )
    args = parser.parse_args()

    races = load_historical_races(Path(args.historical_dir), args.limit)
    base_params = build_base_params()

    search_space = {
        "hard_linear_deg": [0.010, 0.011, 0.012, 0.013, 0.014],
        "hard_quadratic_deg": [0.0004, 0.0005, 0.0006, 0.0007, 0.0008],
        "hard_late_stint_start": [10, 11, 12, 13, 14],
        "hard_late_stint_deg": [0.0003, 0.0004, 0.0005, 0.0006, 0.0008],
    }

    baseline_summary = evaluate_with_params(races, base_params)
    baseline_exact = baseline_summary["exact_matches"]
    baseline_rank_error = sum(
        row["total_abs_rank_error"] for row in baseline_summary["worst_races"]
    )

    print("Baseline result")
    print("===============")
    print(f"Races evaluated:     {baseline_summary['total_races']}")
    print(f"Exact matches:       {baseline_exact}")
    print(f"Exact-match accuracy: {baseline_summary['exact_match_accuracy'] * 100:.2f}%")
    print(f"Total rank error:    {baseline_rank_error}")
    print()

    results = []

    keys = list(search_space.keys())
    for values in itertools.product(*(search_space[key] for key in keys)):
        candidate = dict(zip(keys, values))
        params = apply_candidate(base_params, candidate)
        summary = evaluate_with_params(races, params)

        total_rank_error = sum(
            row["total_abs_rank_error"] for row in summary["worst_races"]
        )

        results.append(
            {
                "candidate": candidate,
                "exact_matches": summary["exact_matches"],
                "exact_match_accuracy": summary["exact_match_accuracy"],
                "total_rank_error": total_rank_error,
            }
        )

    results.sort(
        key=lambda row: (
            -row["exact_matches"],
            row["total_rank_error"],
            row["candidate"]["hard_linear_deg"],
            row["candidate"]["hard_quadratic_deg"],
            row["candidate"]["hard_late_stint_start"],
            row["candidate"]["hard_late_stint_deg"],
        )
    )

    print("Top candidates")
    print("==============")
    for index, row in enumerate(results[: args.show_top], start=1):
        print(f"{index}. exact_matches={row['exact_matches']}, "
              f"accuracy={row['exact_match_accuracy'] * 100:.2f}%, "
              f"total_rank_error={row['total_rank_error']}")
        print(json.dumps(row["candidate"], indent=2))
        print()

    best = results[0]
    best_params = apply_candidate(base_params, best["candidate"])

    print("Best parameter set to try next")
    print("==============================")
    print(json.dumps(best_params["HARD"], indent=2))


if __name__ == "__main__":
    main()