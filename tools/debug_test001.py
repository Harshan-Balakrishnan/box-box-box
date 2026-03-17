import json
import os
import sys

# Add repo root to Python path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solution.race_simulator import simulate_driver

INPUT_FILE = os.path.join(ROOT, "data", "test_cases", "inputs", "test_001.json")

WATCH = {"D019", "D009", "D012", "D004", "D007", "D020", "D016", "D002", "D011"}

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

race_config = data["race_config"]
rows = []

for strategy in data["strategies"].values():
    driver_id = strategy["driver_id"]
    total_time = simulate_driver(race_config, strategy)
    rows.append((total_time, driver_id, strategy))

rows.sort()

print("Predicted order with total times:")
for pos, (total_time, driver_id, strategy) in enumerate(rows, start=1):
    mark = "*" if driver_id in WATCH else " "
    print(
        f"{mark} {pos:2d}. {driver_id}  time={total_time:.6f}  "
        f"start={strategy['starting_tire']}  pits={strategy.get('pit_stops', [])}"
    )

print("\nFocused gaps:")
times = {driver_id: total_time for total_time, driver_id, _ in rows}
pairs = [
    ("D019", "D009"),
    ("D012", "D004"),
    ("D012", "D007"),
    ("D012", "D020"),
    ("D016", "D002"),
    ("D016", "D011"),
]
for a, b in pairs:
    print(f"{a} - {b} = {times[a] - times[b]:.6f}")