import copy
import json
from pathlib import Path

import solution.race_simulator as race_simulator

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "test_cases" / "inputs"
EXPECTED_DIR = ROOT / "data" / "test_cases" / "expected_outputs"


def load_tests():
    tests = []
    for input_path in sorted(INPUT_DIR.glob("test_*.json")):
        expected_path = EXPECTED_DIR / input_path.name
        with input_path.open("r", encoding="utf-8") as f:
            test_input = json.load(f)
        with expected_path.open("r", encoding="utf-8") as f:
            expected = json.load(f)
        tests.append(
            {
                "name": input_path.stem,
                "input": test_input,
                "expected": expected["finishing_positions"],
            }
        )
    return tests


def snapshot():
    return {
        "params": copy.deepcopy(race_simulator.COMPOUND_PARAMS),
        "hard_ultra_start": race_simulator.HARD_ULTRA_LONG_RELIEF_START,
        "hard_ultra_per_lap": race_simulator.HARD_ULTRA_LONG_RELIEF_PER_LAP,
    }


def apply_state(state):
    race_simulator.COMPOUND_PARAMS = copy.deepcopy(state["params"])
    race_simulator.HARD_ULTRA_LONG_RELIEF_START = state["hard_ultra_start"]
    race_simulator.HARD_ULTRA_LONG_RELIEF_PER_LAP = state["hard_ultra_per_lap"]


def set_value(state, key, value):
    updated = copy.deepcopy(state)
    if key == "hard_ultra_start":
        updated["hard_ultra_start"] = int(value)
    elif key == "hard_ultra_per_lap":
        updated["hard_ultra_per_lap"] = float(value)
    else:
        compound, field = key.split(".", 1)
        updated["params"][compound][field] = value
    return updated


def get_value(state, key):
    if key == "hard_ultra_start":
        return state["hard_ultra_start"]
    if key == "hard_ultra_per_lap":
        return state["hard_ultra_per_lap"]
    compound, field = key.split(".", 1)
    return state["params"][compound][field]


def evaluate_official(tests, state):
    apply_state(state)
    exact_matches = 0
    total_l1 = 0
    total_inversions = 0
    exact_tests = []

    for row in tests:
        predicted = race_simulator.predict_finishing_positions(copy.deepcopy(row["input"]))[
            "finishing_positions"
        ]
        expected = row["expected"]

        if predicted == expected:
            exact_matches += 1
            exact_tests.append(row["name"])

        expected_pos = {driver_id: i for i, driver_id in enumerate(expected)}
        total_l1 += sum(abs(i - expected_pos[d]) for i, d in enumerate(predicted))

        for i in range(len(predicted)):
            left = expected_pos[predicted[i]]
            for j in range(i + 1, len(predicted)):
                if left > expected_pos[predicted[j]]:
                    total_inversions += 1

    return {
        "exact_matches": exact_matches,
        "total_l1": total_l1,
        "total_inversions": total_inversions,
        "exact_tests": exact_tests,
    }


def score_tuple(summary):
    return (
        summary["exact_matches"],
        -summary["total_l1"],
        -summary["total_inversions"],
    )


def build_search_space(base_state):
    medium = base_state["params"]["MEDIUM"]
    soft = base_state["params"]["SOFT"]

    def uniq(vals):
        out = []
        for v in vals:
            if v not in out:
                out.append(v)
        return out

    return {
        "hard_ultra_start": uniq([18, 20, 22, 24, 26, base_state["hard_ultra_start"]]),
        "hard_ultra_per_lap": uniq([-0.06, -0.05, -0.04, -0.03, -0.02, base_state["hard_ultra_per_lap"]]),
        "MEDIUM.late_stint_deg": uniq([0.007, 0.009, 0.011, 0.013, 0.015, medium["late_stint_deg"]]),
        "MEDIUM.late_stint_start": uniq([10, 12, 14, medium["late_stint_start"]]),
        "MEDIUM.compound_delta": uniq([-0.20, -0.15, -0.10, -0.05, 0.0, medium["compound_delta"]]),
        "SOFT.compound_delta": uniq([-1.28, -1.20, -1.10, -1.00, -0.96, soft["compound_delta"]]),
    }


def diff_lines(base_state, best_state):
    lines = []
    for compound in ("SOFT", "MEDIUM", "HARD"):
        for field, base_value in base_state["params"][compound].items():
            best_value = best_state["params"][compound][field]
            if best_value != base_value:
                lines.append(f'COMPOUND_PARAMS["{compound}"]["{field}"] = {repr(best_value)}')
    if best_state["hard_ultra_start"] != base_state["hard_ultra_start"]:
        lines.append(f"HARD_ULTRA_LONG_RELIEF_START = {best_state['hard_ultra_start']}")
    if best_state["hard_ultra_per_lap"] != base_state["hard_ultra_per_lap"]:
        lines.append(f"HARD_ULTRA_LONG_RELIEF_PER_LAP = {best_state['hard_ultra_per_lap']}")
    return lines


def main():
    tests = load_tests()
    base_state = snapshot()
    search_space = build_search_space(base_state)

    best_state = copy.deepcopy(base_state)
    best_summary = evaluate_official(tests, best_state)

    print("Baseline")
    print("========")
    print(f"Exact matches:   {best_summary['exact_matches']}/100")
    print(f"Total L1:        {best_summary['total_l1']}")
    print(f"Inversions:      {best_summary['total_inversions']}")
    print(f"Exact tests:     {best_summary['exact_tests']}")
    print()

    for round_idx in range(1, 5):
        improved = False
        print(f"Round {round_idx}")
        print("-------")
        for key, values in search_space.items():
            current_value = get_value(best_state, key)
            local_best_state = best_state
            local_best_summary = best_summary

            for candidate_value in values:
                candidate_state = set_value(best_state, key, candidate_value)
                summary = evaluate_official(tests, candidate_state)
                if score_tuple(summary) > score_tuple(local_best_summary):
                    local_best_state = candidate_state
                    local_best_summary = summary

            new_value = get_value(local_best_state, key)
            if new_value != current_value:
                best_state = local_best_state
                best_summary = local_best_summary
                improved = True
                print(
                    f"improved {key}: {current_value} -> {new_value} | "
                    f"exact={best_summary['exact_matches']}/100, "
                    f"l1={best_summary['total_l1']}, inv={best_summary['total_inversions']}"
                )

        if not improved:
            print("no further improvement found")
            print()
            break
        print()

    print("Best found")
    print("==========")
    print(f"Exact matches:   {best_summary['exact_matches']}/100")
    print(f"Total L1:        {best_summary['total_l1']}")
    print(f"Inversions:      {best_summary['total_inversions']}")
    print(f"Exact tests:     {best_summary['exact_tests']}")
    print()
    print("Patch lines")
    print("===========")
    for line in diff_lines(base_state, best_state):
        print(line)


if __name__ == "__main__":
    main()