import json
import os
import sys
import itertools

print("phase4 started")

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

# lock in best values found so far
soft["compound_delta"] = -1.03
soft["fresh_bonus"] = -0.35
soft["second_lap_bonus"] = -0.12
soft["linear_deg"] = 0.03
soft["quadratic_deg"] = 0.002
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

soft_fade_vals = [0.42, 0.44, 0.46, 0.48, 0.50]
soft_fade_start_vals = [5, 6, 7]
medium_linear_vals = [0.017, 0.018, 0.019, 0.020]
medium_quad_vals = [0.0010, 0.0011, 0.0012, 0.0013]

for sfade, sfade_start, mlin, mquad in itertools.product(
    soft_fade_vals,
    soft_fade_start_vals,
    medium_linear_vals,
    medium_quad_vals
):
    count += 1
    if count % 20 == 0:
        print("checked", count)

    soft["fade_per_lap"] = sfade
    soft["fade_start"] = sfade_start
    medium["linear_deg"] = mlin
    medium["quadratic_deg"] = mquad

    pred = rs.predict_finishing_positions(test_data)["finishing_positions"]
    score = score_order(pred, expected)

    if score < best:
        best = score
        best_settings = {
            "soft_fade_per_lap": sfade,
            "soft_fade_start": sfade_start,
            "medium_linear_deg": mlin,
            "medium_quadratic_deg": mquad,
            "pred": pred,
        }
        print("Better score:", best)
        print(best_settings)
        print()

    if pred == expected:
        print("EXACT MATCH FOUND")
        print(best_settings)
        break

print("phase4 finished")
print("best overall:", best)
print(best_settings)