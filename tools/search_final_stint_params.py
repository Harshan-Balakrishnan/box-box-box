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
    "second_lap_bonus": -0.04,
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
    "fresh_bonus": -0.06,
    "second_lap_bonus": -0.09,
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

    params["MEDIUM"]["compound_delta"] = candidate["medium_compound_delta"]
    params["MEDIUM"]["fresh_bonus"] = candidate["medium_fresh_bonus"]
    params["MEDIUM"]["second_lap_bonus"] = candidate["medium_second_lap_bonus"]
    params["SOFT"]["second_lap_bonus"] = candidate["soft_second_lap_bonus"]

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
    parser = argparse.ArgumentParser(
        description="Search MEDIUM/SOFT final-stint-related parameters using historical races only."
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
        help="How many top candidates to print.",
    )
    args = parser.parse_args()

    races = load_historical_races(Path(args.historical_dir), args.limit)
    base_params = build_base_params()

    search_space = {
        "medium_compound_delta": [-0.20, -0.10, 0.00, 0.10, 0.20],
        "medium_fresh_bonus": [-0.30, -0.24, -0.18, -0.12, -0.06],
        "medium_second_lap_bonus": [-0.12, -0.09, -0.05, -0.02, 0.00],
        "soft_second_lap_bonus": [-0.20, -0.16, -0.12, -0.08, -0.04],
    }

    baseline_summary = evaluate_with_params(races, base_params)
    baseline_exact = baseline_summary["exact_matches"]
    baseline_rank_error = total_rank_error(baseline_summary)

    print("Baseline result")
    print("===============")
    print(f"Races evaluated:      {baseline_summary['total_races']}")
    print(f"Exact matches:        {baseline_exact}")
    print(f"Exact-match accuracy: {baseline_summary['exact_match_accuracy'] * 100:.2f}%")
    print(f"Total rank error:     {baseline_rank_error}")
    print()

    results = []

    keys = list(search_space.keys())
    for values in itertools.product(*(search_space[key] for key in keys)):
        candidate = dict(zip(keys, values))
        params = apply_candidate(base_params, candidate)
        summary = evaluate_with_params(races, params)

        results.append(
            {
                "candidate": candidate,
                "exact_matches": summary["exact_matches"],
                "exact_match_accuracy": summary["exact_match_accuracy"],
                "total_rank_error": total_rank_error(summary),
            }
        )

    results.sort(
        key=lambda row: (
            -row["exact_matches"],
            row["total_rank_error"],
            row["candidate"]["medium_compound_delta"],
            row["candidate"]["medium_fresh_bonus"],
            row["candidate"]["medium_second_lap_bonus"],
            row["candidate"]["soft_second_lap_bonus"],
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
        print()

    best = results[0]
    best_params = apply_candidate(base_params, best["candidate"])

    print("Best parameter set to try next")
    print("==============================")
    print(json.dumps({
        "MEDIUM": {
            "compound_delta": best_params["MEDIUM"]["compound_delta"],
            "fresh_bonus": best_params["MEDIUM"]["fresh_bonus"],
            "second_lap_bonus": best_params["MEDIUM"]["second_lap_bonus"],
        },
        "SOFT": {
            "second_lap_bonus": best_params["SOFT"]["second_lap_bonus"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()