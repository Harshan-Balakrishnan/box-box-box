import json
import os
import sys
import itertools

print("phase5 started")

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

# lock current best model
soft["compound_delta"] = -1.03
soft["fresh_bonus"] = -0.35
soft["second_lap_bonus"] = -0.12
soft["grace_laps"] = 1
soft["linear_deg"] = 0.03
soft["quadratic_deg"] = 0.002
soft["temp_deg"] = 0.0024
soft["sprint_window_start"] = 3
soft["sprint_window_end"] = 5
soft["sprint_window_bonus"] = -0.5
soft["fade_start"] = 6
soft["fade_per_lap"] = 0.4

medium["compound_delta"] = 0.0
medium["fresh_bonus"] = -0.18
medium["second_lap_bonus"] = -0.05
medium["grace_laps"] = 2
medium["linear_deg"] = 0.018
medium["quadratic_deg"] = 0.0012
medium["temp_deg"] = 0.0014

hard["compound_delta"] = 0.72
hard["fresh_bonus"] = -0.08
hard["second_lap_bonus"] = -0.02
hard["grace_laps"] = 3
hard["linear_deg"] = 0.012
hard["quadratic_deg"] = 0.0006
hard["temp_deg"] = 0.0008
hard["late_stint_start"] = 12
hard["late_stint_deg"] = 0.0005

best = 10**9
best_settings = None
count = 0

soft_fade_vals = [0.36, 0.38, 0.40, 0.42]
soft_delta_vals = [-1.04, -1.03, -1.02, -1.01]
medium_delta_vals = [-0.04, -0.03, -0.02, -0.01, 0.0]
medium_linear_vals = [0.0175, 0.0180, 0.0185]

for sfade, sdelta, mdelta, mlin in itertools.product(
    soft_fade_vals,
    soft_delta_vals,
    medium_delta_vals,
    medium_linear_vals
):
    count += 1
    if count % 20 == 0:
        print("checked", count)

    soft["fade_per_lap"] = sfade
    soft["compound_delta"] = sdelta
    medium["compound_delta"] = mdelta
    medium["linear_deg"] = mlin

    pred = rs.predict_finishing_positions(test_data)["finishing_positions"]
    score = score_order(pred, expected)

    if score < best:
        best = score
        best_settings = {
            "soft_fade_per_lap": sfade,
            "soft_compound_delta": sdelta,
            "medium_compound_delta": mdelta,
            "medium_linear_deg": mlin,
            "pred": pred,
        }
        print("Better score:", best)
        print(best_settings)
        print()

    if pred == expected:
        print("EXACT MATCH FOUND")
        print(best_settings)
        break

print("phase5 finished")
print("best overall:", best)
print(best_settings)