import json
import subprocess

input_file = "data/test_cases/inputs/test_001.json"
expected_file = "data/test_cases/expected_outputs/test_001.json"

with open(expected_file, "r") as f:
    expected = json.load(f)

result = subprocess.run(
    ["py", "solution/race_simulator.py"],
    stdin=open(input_file, "r"),
    capture_output=True,
    text=True
)

predicted = json.loads(result.stdout)

print("EXPECTED:")
print(expected["finishing_positions"])
print()
print("PREDICTED:")
print(predicted["finishing_positions"])
print()
print("DIFFERENCES:")
for i, (e, p) in enumerate(zip(expected["finishing_positions"], predicted["finishing_positions"]), start=1):
    marker = "OK" if e == p else "WRONG"
    print(f"{i:2d}. expected={e} predicted={p} -> {marker}")