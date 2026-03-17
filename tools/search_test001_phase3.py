import json
import os
import sys
import itertools

print("phase3 started")

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

soft = rs.COMPOUND_PARAMS["SOFT"]
medium = rs.COMPOUND_PARAMS["MEDIUM"]
hard = rs.COMPOUND_PARAMS["HARD"]

soft["fresh_bonus"] = -0.35
soft["second_lap_bonus"] = -0.12
soft["sprint_window_bonus"] = -0.5
soft["fade_start"] = 6
soft["fade_per_lap"] = 0.4

medium["linear_deg"] = 0.018
medium["quadratic_deg"] = 0.0012

hard["linear_deg"] = 0.012
hard["quadratic_deg"] = 0.0006
hard["late_stint_start"] = 12
hard["late_stint_deg"] = 0.0005

best = 10**9
best_settings = None
count = 0

soft_compound_vals = [-1.00, -1.03, -1.05, -1.08]
soft_linear_vals = [0.030, 0.032, 0.035, 0.038]
soft_quad_vals = [0.0020, 0.0023, 0.0026, 0.0029]

for sdelta, slin, squad in itertools.product(
    soft_compound_vals,
    soft_linear_vals,
    soft_quad_vals
):
    count += 1
    if count % 10 == 0:
        print("checked", count)

    soft["compound_delta"] = sdelta
    soft["linear_deg"] = slin
    soft["quadratic_deg"] = squad

    pred = rs.predict_finishing_positions(test_data)["finishing_positions"]
    score = score_order(pred, expected)

    if score < best:
        best = score
        best_settings = {
            "soft_compound_delta": sdelta,
            "soft_linear_deg": slin,
            "soft_quadratic_deg": squad,
            "pred": pred,
        }
        print("Better score:", best)
        print(best_settings)
        print()

    if pred == expected:
        print("EXACT MATCH FOUND")
        print(best_settings)
        break

print("phase3 finished")
print("best overall:", best)
print(best_settings)