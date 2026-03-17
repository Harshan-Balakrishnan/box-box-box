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
    params["HARD"]["compound_delta"] = candidate["hard_compound_delta"]
    params["HARD"]["linear_deg"] = candidate["hard_linear_deg"]
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
    parser = argparse.ArgumentParser(description="Search SOFT->HARD crossover parameters.")
    parser.add_argument("--historical-dir", default="data/historical_races")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--show-top", type=int, default=10)
    args = parser.parse_args()

    races = load_historical_races(Path(args.historical_dir), args.limit)
    base_params = build_base_params()

    search_space = {
        "soft_compound_delta": [-1.02, -1.00, -0.98, -0.96],
        "soft_linear_deg": [0.026, 0.028, 0.030],
        "soft_quadratic_deg": [0.0020, 0.0022, 0.0024],
        "soft_fade_start": [5, 6, 7],
        "soft_fade_per_lap": [0.20, 0.25, 0.30],
        "hard_compound_delta": [0.68, 0.70, 0.72, 0.74],
        "hard_linear_deg": [0.009, 0.010, 0.011],
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
    print("Best crossover candidate")
    print("========================")
    print(json.dumps(best["candidate"], indent=2))


if __name__ == "__main__":
    main()