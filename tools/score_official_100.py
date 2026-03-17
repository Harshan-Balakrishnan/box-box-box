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


def main():
    tests = load_tests()

    exact_matches = 0
    failures = []

    for row in tests:
        predicted = race_simulator.predict_finishing_positions(copy.deepcopy(row["input"]))[
            "finishing_positions"
        ]
        expected = row["expected"]

        if predicted == expected:
            exact_matches += 1
        else:
            failures.append(row["name"])

    print("Official 100 score")
    print("==================")
    print(f"Exact matches: {exact_matches}/100")
    print("Passed tests:")
    print([row["name"] for row in tests if row["name"] not in failures])
    print("Failed tests:")
    print(failures)


if __name__ == "__main__":
    main()