import json
import os
import sys
import itertools

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import solution.race_simulator as rs


INPUT_FILE = os.path.join(ROOT, "data", "test_cases", "inputs", "test_001.json")
EXPECTED_FILE = os.path.join(ROOT, "data", "test_cases", "expected_outputs", "test_001.json")


def predict_order(data):
    result = rs.predict_finishing_positions(data)
    return result["finishing_positions"]


def score_order(pred, expected):
    # lower is better
    pos = {driver: i for i, driver in enumerate(expected)}
    return sum(abs(i - pos[d]) for i, d in enumerate(pred))


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

with open(EXPECTED_FILE, "r", encoding="utf-8") as f:
    expected = json.load(f)["finishing_positions"]

best = None
best_settings = None

soft = rs.COMPOUND_PARAMS["SOFT"]

fresh_values = [-0.35, -0.40, -0.45]
second_values = [-0.12, -0.16, -0.20]
sprint_values = [-0.20, -0.30, -0.40, -0.50, -0.60]
fade_start_values = [6, 7, 8]
fade_per_lap_values = [0.20, 0.30, 0.40, 0.50, 0.65]

for fresh_bonus, second_lap_bonus, sprint_bonus, fade_start, fade_per_lap in itertools.product(
    fresh_values,
    second_values,
    sprint_values,
    fade_start_values,
    fade_per_lap_values
):
    soft["fresh_bonus"] = fresh_bonus
    soft["second_lap_bonus"] = second_lap_bonus
    soft["sprint_window_bonus"] = sprint_bonus
    soft["fade_start"] = fade_start
    soft["fade_per_lap"] = fade_per_lap

    pred = predict_order(test_data)
    score = score_order(pred, expected)

    if best is None or score < best:
        best = score
        best_settings = {
            "fresh_bonus": fresh_bonus,
            "second_lap_bonus": second_lap_bonus,
            "sprint_window_bonus": sprint_bonus,
            "fade_start": fade_start,
            "fade_per_lap": fade_per_lap,
            "pred": pred,
        }
        print("Better score:", best)
        print(best_settings)
        print()

    if pred == expected:
        print("EXACT MATCH FOUND")
        print(best_settings)
        break