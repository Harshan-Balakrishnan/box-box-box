import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb


COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
COMPOUND_TO_IDX = {c: i for i, c in enumerate(COMPOUNDS)}

# Stable 14/100 simulator baseline
SIM_PARAMS = {
    "SOFT": {
        "compound_delta": -0.96,
        "fresh_bonus": -0.31,
        "second_lap_bonus": 0.0,
        "grace_laps": 1,
        "linear_deg": 0.03,
        "quadratic_deg": 0.0022,
        "temp_deg": 0.0024,
        "sprint_window_start": 3,
        "sprint_window_end": 5,
        "sprint_window_bonus": -0.5,
        "fade_start": 7,
        "fade_per_lap": 0.2,
    },
    "MEDIUM": {
        "compound_delta": -0.15,
        "fresh_bonus": -0.03,
        "second_lap_bonus": 0.0,
        "grace_laps": 2,
        "linear_deg": 0.016,
        "quadratic_deg": 0.001,
        "temp_deg": 0.0014,
        "late_stint_start": 12,
        "late_stint_deg": 0.013,
    },
    "HARD": {
        "compound_delta": 0.68,
        "fresh_bonus": -0.08,
        "second_lap_bonus": -0.02,
        "grace_laps": 3,
        "linear_deg": 0.01,
        "quadratic_deg": 0.0005,
        "temp_deg": 0.0008,
        "late_stint_start": 13,
        "late_stint_deg": 0.0003,
    },
}
HARD_ULTRA_LONG_RELIEF_START = 20
HARD_ULTRA_LONG_RELIEF_PER_LAP = -0.05


def normalize_compound(compound: str) -> str:
    compound = str(compound).upper()
    return compound if compound in COMPOUND_TO_IDX else "MEDIUM"


def extract_stints(strategy: dict, total_laps: int):
    current = normalize_compound(strategy["starting_tire"])
    last_start = 1
    stints = []

    for stop in strategy.get("pit_stops", []):
        lap = int(stop["lap"])
        stints.append((current, lap - last_start + 1))
        current = normalize_compound(stop["to_tire"])
        last_start = lap + 1

    stints.append((current, total_laps - last_start + 1))
    return stints


def age_bucket(age: int) -> int:
    if age <= 3:
        return 0
    if age <= 6:
        return 1
    if age <= 10:
        return 2
    if age <= 15:
        return 3
    if age <= 22:
        return 4
    if age <= 32:
        return 5
    return 6


def featurize_race_driver(race: dict, strategy: dict) -> np.ndarray:
    rc = race["race_config"]
    total_laps = int(rc["total_laps"])
    base_lap_time = float(rc["base_lap_time"])
    pit_lane_time = float(rc["pit_lane_time"])
    track_temp = float(rc["track_temp"])
    temp_offset = track_temp - 30.0

    stints = extract_stints(strategy, total_laps)
    pit_count = len(strategy.get("pit_stops", []))

    feats = []
    feats.extend([
        total_laps,
        base_lap_time,
        pit_lane_time,
        track_temp,
        temp_offset,
        pit_count,
        pit_count * pit_lane_time,
    ])

    start = normalize_compound(strategy["starting_tire"])
    for c in COMPOUNDS:
        feats.append(1.0 if start == c else 0.0)

    used_sequence = [c for c, _ in stints][:3]
    for slot in range(3):
        cur = used_sequence[slot] if slot < len(used_sequence) else "NONE"
        for c in COMPOUNDS + ["NONE"]:
            feats.append(1.0 if cur == c else 0.0)

    compound_laps = defaultdict(float)
    compound_stints = defaultdict(float)
    compound_age1 = defaultdict(float)
    compound_age2 = defaultdict(float)
    compound_age3 = defaultdict(float)
    compound_temp_age1 = defaultdict(float)
    compound_temp_age2 = defaultdict(float)
    compound_bucket_counts = defaultdict(lambda: [0.0] * 7)
    compound_last_stint_len = defaultdict(float)
    compound_max_stint = defaultdict(float)

    for compound, stint_len in stints:
        compound_laps[compound] += stint_len
        compound_stints[compound] += 1
        compound_last_stint_len[compound] = stint_len
        compound_max_stint[compound] = max(compound_max_stint[compound], stint_len)

        for age in range(1, stint_len + 1):
            compound_age1[compound] += age
            compound_age2[compound] += age * age
            compound_age3[compound] += age * age * age
            compound_temp_age1[compound] += temp_offset * age
            compound_temp_age2[compound] += temp_offset * age * age
            compound_bucket_counts[compound][age_bucket(age)] += 1.0

    for c in COMPOUNDS:
        laps = compound_laps[c]
        st_count = compound_stints[c]
        feats.extend([
            laps,
            laps / max(1, total_laps),
            st_count,
            compound_age1[c],
            compound_age2[c],
            compound_age3[c],
            compound_temp_age1[c],
            compound_temp_age2[c],
            compound_last_stint_len[c],
            compound_max_stint[c],
            (laps / st_count) if st_count else 0.0,
        ])
        feats.extend(compound_bucket_counts[c])

    stint_lengths = [length for _, length in stints]
    feats.extend([
        len(stints),
        min(stint_lengths),
        max(stint_lengths),
        sum(stint_lengths) / len(stint_lengths),
        sum(x * x for x in stint_lengths),
    ])

    transitions = ["->".join([stints[i][0], stints[i + 1][0]]) for i in range(len(stints) - 1)]
    all_transitions = [
        "SOFT->MEDIUM", "SOFT->HARD",
        "MEDIUM->SOFT", "MEDIUM->HARD",
        "HARD->SOFT", "HARD->MEDIUM"
    ]
    for t in all_transitions:
        feats.append(float(transitions.count(t)))

    return np.array(feats, dtype=np.float32)


def lap_time(base_lap_time, compound, tire_age, track_temp):
    compound = normalize_compound(compound)
    params = SIM_PARAMS[compound]

    effective_age = max(0, tire_age - params["grace_laps"])
    temp_offset = float(track_temp) - 30.0

    base_degradation = (
        params["linear_deg"] * effective_age
        + params["quadratic_deg"] * (effective_age ** 2)
        + params["temp_deg"] * temp_offset * effective_age
    )

    late_stint_penalty = 0.0
    if compound in ("HARD", "MEDIUM"):
        late_stint_age = max(0, effective_age - params["late_stint_start"])
        late_stint_penalty = params["late_stint_deg"] * (late_stint_age ** 2)

        if compound == "HARD" and tire_age >= HARD_ULTRA_LONG_RELIEF_START:
            ultra_long_laps = tire_age - HARD_ULTRA_LONG_RELIEF_START + 1
            late_stint_penalty += HARD_ULTRA_LONG_RELIEF_PER_LAP * ultra_long_laps

    fresh_bonus = 0.0
    if tire_age == 1:
        fresh_bonus = params["fresh_bonus"]
    elif tire_age == 2:
        fresh_bonus = params["second_lap_bonus"]

    soft_stint_adjustment = 0.0
    if compound == "SOFT":
        if params["sprint_window_start"] <= tire_age <= params["sprint_window_end"]:
            soft_stint_adjustment += params["sprint_window_bonus"]

        fade_laps = max(0, tire_age - params["fade_start"] + 1)
        soft_stint_adjustment += params["fade_per_lap"] * fade_laps

    return (
        float(base_lap_time)
        + params["compound_delta"]
        + fresh_bonus
        + base_degradation
        + late_stint_penalty
        + soft_stint_adjustment
    )


def simulate_driver(race_config, strategy):
    total_laps = int(race_config["total_laps"])
    base_lap_time_val = float(race_config["base_lap_time"])
    pit_lane_time = float(race_config["pit_lane_time"])
    track_temp = float(race_config["track_temp"])

    current_tire = normalize_compound(strategy["starting_tire"])
    tire_age = 0
    total_time = 0.0

    pit_map = {}
    for stop in strategy.get("pit_stops", []):
        pit_map[int(stop["lap"])] = stop

    for lap in range(1, total_laps + 1):
        tire_age += 1
        total_time += lap_time(base_lap_time_val, current_tire, tire_age, track_temp)

        if lap in pit_map:
            stop = pit_map[lap]
            total_time += pit_lane_time
            current_tire = normalize_compound(stop["to_tire"])
            tire_age = 0

    return total_time


def ranks_from_scores(scores, higher_is_better=True):
    order = np.argsort(scores)
    if higher_is_better:
        order = order[::-1]
    rank = np.empty(len(scores), dtype=np.int32)
    for i, idx in enumerate(order):
        rank[idx] = i
    return rank


def load_model():
    repo_root = Path(__file__).resolve().parents[1]
    model_dir = repo_root / "solution" / "trained_model"
    model_path = model_dir / "ranker.json"
    meta_path = model_dir / "meta.json"

    if not model_path.exists() or not meta_path.exists():
        return None, None

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    meta = json.loads(meta_path.read_text())
    return booster, meta


def predict_finishing_positions(data):
    driver_ids = []
    sim_times = []
    X_rows = []
    grid_positions = []

    for pos_key, strategy in data["strategies"].items():
        driver_ids.append(strategy["driver_id"])
        sim_times.append(simulate_driver(data["race_config"], strategy))
        X_rows.append(featurize_race_driver(data, strategy))
        grid_positions.append(int(str(pos_key).replace("pos", "")))

    booster, meta = load_model()
    if booster is None:
        rows = list(zip(sim_times, grid_positions, driver_ids))
        rows.sort(key=lambda item: (item[0], item[1]))
        return {
            "race_id": data["race_id"],
            "finishing_positions": [driver_id for _, _, driver_id in rows],
        }

    X = np.stack(X_rows)
    dtest = xgb.DMatrix(X)
    model_scores = booster.predict(dtest)

    # Fixed blend instead of trusting alpha=1.0
    alpha = 0.35

    sim_rank = ranks_from_scores(np.array(sim_times), higher_is_better=False)
    model_rank = ranks_from_scores(model_scores, higher_is_better=True)

    hybrid_rank = alpha * model_rank + (1.0 - alpha) * sim_rank

    rows = list(zip(hybrid_rank, sim_times, grid_positions, driver_ids))
    rows.sort(key=lambda item: (item[0], item[1], item[2]))

    return {
        "race_id": data["race_id"],
        "finishing_positions": [driver_id for _, _, _, driver_id in rows],
    }


def main():
    data = json.load(sys.stdin)
    output = predict_finishing_positions(data)
    sys.stdout.write(json.dumps(output))


if __name__ == "__main__":
    main()