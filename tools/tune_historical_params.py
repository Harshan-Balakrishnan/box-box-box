import argparse
import copy
import itertools
import json
from pathlib import Path

import solution.race_simulator as race_simulator


REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_DIR = REPO_ROOT / "data" / "historical_races"
PARAMS_PATH = REPO_ROOT / "solution" / "model_params.json"


def load_historical_races(limit=None):
    races = []
    files = sorted(HISTORICAL_DIR.glob("*.json"))
    for path in files:
        batch = json.loads(path.read_text())
        for race in batch:
            races.append(race)
            if limit is not None and len(races) >= limit:
                return races
    return races


def rank_error(predicted, expected):
    expected_pos = {driver_id: i for i, driver_id in enumerate(expected)}
    return sum(abs(i - expected_pos[driver_id]) for i, driver_id in enumerate(predicted))


def score_races(races, params):
    exact = 0
    total_rank_error = 0

    for race in races:
        predicted = race_simulator.predict_finishing_positions(race, params=params)["finishing_positions"]
        expected = race["finishing_positions"]

        if predicted == expected:
            exact += 1

        total_rank_error += rank_error(predicted, expected)

    return exact, total_rank_error


def set_nested(params, path, value):
    cur = params
    for key in path[:-1]:
        cur = cur[key]
    cur[path[-1]] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--max-combos", type=int, default=500)
    args = parser.parse_args()

    races = load_historical_races(limit=args.limit)

    if PARAMS_PATH.exists():
        base_params = json.loads(PARAMS_PATH.read_text())
    else:
        base_params = copy.deepcopy(race_simulator.DEFAULT_PARAMS)

    base_exact, base_rank_error = score_races(races, base_params)
    print("Baseline")
    print("========")
    print(f"Races:               {len(races)}")
    print(f"Exact matches:       {base_exact}")
    print(f"Exact-match accuracy: {100.0 * base_exact / max(1, len(races)):.2f}%")
    print(f"Total rank error:    {base_rank_error}")
    print()

    search_space = {
        ("pit_on_lap_uses_new_tire",): [False, True],
        ("tie_break_with_grid",): [False, True],

        ("SOFT", "compound_delta"): [-1.10, -1.04, -0.98, -0.92],
        ("SOFT", "fresh_bonus"): [-0.35, -0.28, -0.20],
        ("SOFT", "second_lap_bonus"): [-0.08, -0.02, 0.00],
        ("SOFT", "linear_deg"): [0.024, 0.028, 0.032],
        ("SOFT", "quadratic_deg"): [0.0018, 0.0022, 0.0026],
        ("SOFT", "sprint_window_start"): [2, 3],
        ("SOFT", "sprint_window_end"): [4, 5],
        ("SOFT", "sprint_window_bonus"): [-0.40, -0.30, -0.20],
        ("SOFT", "fade_start"): [4, 5, 6, 7],
        ("SOFT", "fade_per_lap"): [0.14, 0.18, 0.20, 0.24],
        ("SOFT", "planned_stint_len_coeff"): [0.006, 0.012, 0.018],
        ("SOFT", "late_stint_cliff_start"): [6, 7, 8],
        ("SOFT", "late_stint_cliff_per_lap"): [0.05, 0.10, 0.15],

        ("MEDIUM", "compound_delta"): [-0.10, -0.05, 0.00, 0.05],
        ("MEDIUM", "fresh_bonus"): [-0.08, -0.05, -0.02],
        ("MEDIUM", "second_lap_bonus"): [-0.05, -0.03, 0.0],
        ("MEDIUM", "linear_deg"): [0.012, 0.016, 0.020],
        ("MEDIUM", "quadratic_deg"): [0.0006, 0.0010, 0.0014],
        ("MEDIUM", "late_stint_start"): [10, 12, 14],
        ("MEDIUM", "late_stint_deg"): [0.0, 0.006, 0.012],
        ("MEDIUM", "planned_stint_len_coeff"): [0.0, 0.006, 0.012],
        ("MEDIUM", "late_stint_cliff_start"): [14, 16, 18],
        ("MEDIUM", "late_stint_cliff_per_lap"): [0.0, 0.03, 0.06],

        ("HARD", "compound_delta"): [0.55, 0.68, 0.80],
        ("HARD", "fresh_bonus"): [-0.12, -0.08, -0.04],
        ("HARD", "second_lap_bonus"): [-0.05, -0.03, 0.0],
        ("HARD", "linear_deg"): [0.006, 0.010, 0.014],
        ("HARD", "quadratic_deg"): [0.0002, 0.0005, 0.0008],
        ("HARD", "late_stint_start"): [11, 13, 16],
        ("HARD", "late_stint_deg"): [0.0, 0.0003, 0.0008],
        ("HARD", "planned_stint_len_coeff"): [-0.009, -0.003, 0.0],
        ("HARD", "late_stint_cliff_start"): [18, 22, 26],
        ("HARD", "late_stint_cliff_per_lap"): [0.0, 0.01, 0.03],

        ("GLOBAL", "hard_ultra_long_relief_start"): [16, 20, 24],
        ("GLOBAL", "hard_ultra_long_relief_per_lap"): [-0.10, -0.05, 0.0],

        ("GLOBAL", "transition_bonus_soft_to_hard"): [-0.20, -0.10, 0.0],
        ("GLOBAL", "transition_bonus_medium_to_hard"): [-0.05, 0.00, 0.05, 0.10],
        ("GLOBAL", "transition_bonus_hard_to_soft"): [-0.25, -0.18, -0.10],
        ("GLOBAL", "transition_bonus_hard_to_medium"): [-0.12, -0.08, -0.04],
        ("GLOBAL", "transition_bonus_soft_to_medium"): [-0.04, 0.00, 0.04],
        ("GLOBAL", "transition_bonus_medium_to_soft"): [-0.14, -0.08, -0.02],
    }

    parameter_groups = [
        [
            ("pit_on_lap_uses_new_tire",),
            ("tie_break_with_grid",),
        ],
        [
            ("SOFT", "compound_delta"),
            ("SOFT", "fresh_bonus"),
            ("SOFT", "fade_start"),
            ("SOFT", "fade_per_lap"),
        ],
        [
            ("MEDIUM", "compound_delta"),
            ("MEDIUM", "late_stint_start"),
            ("MEDIUM", "late_stint_deg"),
            ("MEDIUM", "planned_stint_len_coeff"),
        ],
        [
            ("HARD", "compound_delta"),
            ("HARD", "late_stint_start"),
            ("HARD", "late_stint_deg"),
            ("GLOBAL", "hard_ultra_long_relief_start"),
            ("GLOBAL", "hard_ultra_long_relief_per_lap"),
        ],
        [
            ("GLOBAL", "transition_bonus_soft_to_hard"),
            ("GLOBAL", "transition_bonus_medium_to_hard"),
            ("GLOBAL", "transition_bonus_hard_to_soft"),
            ("GLOBAL", "transition_bonus_hard_to_medium"),
        ],
        [
            ("SOFT", "planned_stint_len_coeff"),
            ("SOFT", "late_stint_cliff_start"),
            ("SOFT", "late_stint_cliff_per_lap"),
            ("MEDIUM", "late_stint_cliff_start"),
            ("MEDIUM", "late_stint_cliff_per_lap"),
        ],
        [
            ("SOFT", "quadratic_deg"),
            ("MEDIUM", "quadratic_deg"),
            ("HARD", "quadratic_deg"),
        ],
    ]

    best_params = copy.deepcopy(base_params)
    best_exact = base_exact
    best_rank_error = base_rank_error
    combos_tried = 0

    for group in parameter_groups:
        value_lists = [search_space[g] for g in group]

        for combo in itertools.product(*value_lists):
            combos_tried += 1
            if combos_tried > args.max_combos:
                break

            candidate = copy.deepcopy(best_params)
            for path, value in zip(group, combo):
                set_nested(candidate, path, value)

            exact, total_rank_error = score_races(races, candidate)

            improved = False
            if exact > best_exact:
                improved = True
            elif exact == best_exact and total_rank_error < best_rank_error:
                improved = True

            if improved:
                best_params = copy.deepcopy(candidate)
                best_exact = exact
                best_rank_error = total_rank_error

                print("New best found")
                print("==============")
                print(json.dumps(best_params, indent=2))
                print(f"Exact matches:       {best_exact}")
                print(f"Exact-match accuracy: {100.0 * best_exact / max(1, len(races)):.2f}%")
                print(f"Total rank error:    {best_rank_error}")
                print()

        if combos_tried > args.max_combos:
            break

    PARAMS_PATH.write_text(json.dumps(best_params, indent=2))

    print("Best parameters found")
    print("=====================")
    print(json.dumps(best_params, indent=2))
    print(f"Exact matches:       {best_exact}")
    print(f"Exact-match accuracy: {100.0 * best_exact / max(1, len(races)):.2f}%")
    print(f"Total rank error:    {best_rank_error}")
    print()
    print(f"Saved to {PARAMS_PATH}")


if __name__ == "__main__":
    main()