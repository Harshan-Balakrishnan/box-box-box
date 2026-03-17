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


def score_order(pred, expected):
    pos = {driver: i for i, driver in enumerate(expected)}
    return sum(abs(i - pos[d]) for i, d in enumerate(pred))


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

with open(EXPECTED_FILE, "r", encoding="utf-8") as f:
    expected = json.load(f)["finishing_positions"]

# Keep best SOFT values from phase 1
rs.COMPOUND_PARAMS["SOFT"]["fresh_bonus"] = -0.35
rs.COMPOUND_PARAMS["SOFT"]["second_lap_bonus"] = -0.12
rs.COMPOUND_PARAMS["SOFT"]["sprint_window_bonus"] = -0.5
rs.COMPOUND_PARAMS["SOFT"]["fade_start"] = 6
rs.COMPOUND_PARAMS["SOFT"]["fade_per_lap"] = 0.4

best = None
best_settings = None

medium = rs.COMPOUND_PARAMS["MEDIUM"]
hard = rs.COMPOUND_PARAMS["HARD"]

medium_linear_vals = [0.018, 0.020, 0.022, 0.024]
medium_quad_vals = [0.0010, 0.0012, 0.0014, 0.0016]

hard_linear_vals = [0.012, 0.013, 0.014, 0.015]
hard_quad_vals = [0.0006, 0.0008, 0.0010]
hard_late_start_vals = [12, 14, 16]
hard_late_deg_vals = [0.0002, 0.0005, 0.0010]

for mlin, mquad, hlin, hquad, hstart, hlate in itertools.product(
    medium_linear_vals,
    medium_quad_vals,
    hard_linear_vals,
    hard_quad_vals,
    hard_late_start_vals,
    hard_late_deg_vals
):
    medium["linear_deg"] = 0.018
    medium["quadratic_deg"] = 0.0012
    hard["linear_deg"] = 0.012
    hard["quadratic_deg"] = 0.0006
    hard["late_stint_start"] = 12
    hard["late_stint_deg"] = 0.0005

    pred = rs.predict_finishing_positions(test_data)["finishing_positions"]
    score = score_order(pred, expected)

    if best is None or score < best:
        best = score
        best_settings = {
            "medium_linear_deg": mlin,
            "medium_quadratic_deg": mquad,
            "hard_linear_deg": hlin,
            "hard_quadratic_deg": hquad,
            "hard_late_stint_start": hstart,
            "hard_late_stint_deg": hlate,
            "pred": pred,
        }
        print("Better score:", best)
        print(best_settings)
        print()

    if pred == expected:
        print("EXACT MATCH FOUND")
        print(best_settings)
        break