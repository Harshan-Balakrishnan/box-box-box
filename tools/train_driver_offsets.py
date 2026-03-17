import argparse
import json
from collections import defaultdict
from pathlib import Path

import solution.race_simulator as race_simulator


REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_DIR = REPO_ROOT / "data" / "historical_races"
OUTPUT_PATH = REPO_ROOT / "solution" / "driver_offsets.json"


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
    err = 0
    for i, driver_id in enumerate(predicted):
        err += abs(i - expected_pos[driver_id])
    return err


def predict_with_offsets(race, offsets):
    race_config = race["race_config"]
    rows = []

    for pos_key, strategy in race["strategies"].items():
        driver_id = strategy["driver_id"]
        base_time = race_simulator.simulate_driver(race_config, strategy)
        corrected_time = base_time + float(offsets.get(driver_id, 0.0))
        grid_position = int(str(pos_key).replace("pos", ""))
        rows.append((corrected_time, grid_position, driver_id))

    rows.sort(key=lambda item: (item[0], item[1]))
    return [driver_id for _, _, driver_id in rows]


def evaluate(races, offsets):
    exact = 0
    total_rank_error = 0

    for race in races:
        predicted = predict_with_offsets(race, offsets)
        expected = race["finishing_positions"]

        if predicted == expected:
            exact += 1

        total_rank_error += rank_error(predicted, expected)

    return exact, total_rank_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()

    races = load_historical_races(limit=args.limit)

    driver_residuals = defaultdict(list)

    for race in races:
        race_config = race["race_config"]
        rows = []

        for pos_key, strategy in race["strategies"].items():
            driver_id = strategy["driver_id"]
            total_time = race_simulator.simulate_driver(race_config, strategy)
            grid_position = int(str(pos_key).replace("pos", ""))
            rows.append((total_time, grid_position, driver_id))

        rows.sort(key=lambda item: (item[0], item[1]))
        predicted_order = [driver_id for _, _, driver_id in rows]
        actual_order = race["finishing_positions"]

        predicted_pos = {driver_id: i for i, driver_id in enumerate(predicted_order)}
        actual_pos = {driver_id: i for i, driver_id in enumerate(actual_order)}

        for driver_id in actual_pos:
            residual = predicted_pos[driver_id] - actual_pos[driver_id]
            driver_residuals[driver_id].append(residual)

    avg_residual = {
        driver_id: sum(values) / len(values)
        for driver_id, values in driver_residuals.items()
        if values
    }

    scales = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
    best_scale = None
    best_offsets = None
    best_exact = -1
    best_rank_error = None

    for scale in scales:
        offsets = {
            driver_id: -scale * residual
            for driver_id, residual in avg_residual.items()
        }

        exact, total_rank_error = evaluate(races, offsets)

        improved = False
        if exact > best_exact:
            improved = True
        elif exact == best_exact and (best_rank_error is None or total_rank_error < best_rank_error):
            improved = True

        if improved:
            best_scale = scale
            best_offsets = offsets
            best_exact = exact
            best_rank_error = total_rank_error

    print("Best driver offset scale")
    print("========================")
    print(best_scale)
    print()
    print("Training score")
    print("==============")
    print(f"Races:               {len(races)}")
    print(f"Exact matches:       {best_exact}")
    print(f"Exact-match accuracy: {100.0 * best_exact / max(1, len(races)):.2f}%")
    print(f"Total rank error:    {best_rank_error}")
    print()

    ordered = dict(sorted(best_offsets.items()))
    print("Driver offsets")
    print("==============")
    print(json.dumps(ordered, indent=2))

    OUTPUT_PATH.write_text(json.dumps(ordered, indent=2))
    print()
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()