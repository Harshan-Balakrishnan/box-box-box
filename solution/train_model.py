import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb


COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
COMPOUND_TO_IDX = {c: i for i, c in enumerate(COMPOUNDS)}

# Stable simulator baseline
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


def rank_error(predicted, expected):
    expected_pos = {driver_id: i for i, driver_id in enumerate(expected)}
    return sum(abs(i - expected_pos[d]) for i, d in enumerate(predicted))


def load_all_races(repo_root: Path):
    historical_dir = repo_root / "data" / "historical_races"
    races = []
    for path in sorted(historical_dir.glob("*.json")):
        races.extend(json.loads(path.read_text()))
    return races


def build_matrix_and_groups(races):
    X_rows = []
    y_rows = []
    groups = []

    for race in races:
        rank_map = {
            driver_id: 20 - idx
            for idx, driver_id in enumerate(race["finishing_positions"])
        }

        rows = []
        labels = []
        for _, strategy in race["strategies"].items():
            rows.append(featurize_race_driver(race, strategy))
            labels.append(rank_map[strategy["driver_id"]])

        X_rows.extend(rows)
        y_rows.extend(labels)
        groups.append(len(rows))

    X = np.stack(X_rows)
    y = np.array(y_rows, dtype=np.float32)
    return X, y, groups


def choose_blend_weight(model, valid_races):
    best_alpha = 0.65
    best_exact = -1
    best_rank_err = 10**18

    for alpha in [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]:
        exact = 0
        total_rank_err = 0

        for race in valid_races:
            driver_ids = []
            sim_times = []
            X_rows = []

            for pos_key, strategy in race["strategies"].items():
                driver_ids.append(strategy["driver_id"])
                sim_times.append(simulate_driver(race["race_config"], strategy))
                X_rows.append(featurize_race_driver(race, strategy))

            X = np.stack(X_rows)
            dvalid = xgb.DMatrix(X)
            model_scores = model.predict(dvalid)

            sim_rank = ranks_from_scores(np.array(sim_times), higher_is_better=False)
            model_rank = ranks_from_scores(model_scores, higher_is_better=True)

            hybrid = alpha * model_rank + (1.0 - alpha) * sim_rank
            order = np.argsort(hybrid)
            predicted = [driver_ids[i] for i in order]
            expected = race["finishing_positions"]

            if predicted == expected:
                exact += 1
            total_rank_err += rank_error(predicted, expected)

        if exact > best_exact or (exact == best_exact and total_rank_err < best_rank_err):
            best_alpha = alpha
            best_exact = exact
            best_rank_err = total_rank_err

    return best_alpha, best_exact, best_rank_err


def main():
    repo_root = Path(__file__).resolve().parents[1]
    model_dir = repo_root / "solution" / "trained_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    races = load_all_races(repo_root)
    split_idx = max(100, int(len(races) * 0.9))
    train_races = races[:split_idx]
    valid_races = races[split_idx:]

    X_train, y_train, groups_train = build_matrix_and_groups(train_races)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(groups_train)

    params = {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg",
        "eta": 0.05,
        "max_depth": 7,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 2.0,
        "min_child_weight": 3.0,
        "seed": 42,
        "tree_method": "hist",
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=700,
        verbose_eval=False,
    )

    alpha, valid_exact, valid_rank_err = choose_blend_weight(booster, valid_races)

    booster.save_model(model_dir / "ranker.json")
    meta = {
        "feature_dim": int(X_train.shape[1]),
        "blend_alpha_model_rank": alpha,
        "validation_exact_matches": int(valid_exact),
        "validation_rank_error": int(valid_rank_err),
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Model saved to: {model_dir}")
    print(f"Chosen alpha: {alpha}")
    print(f"Validation exact matches: {valid_exact}")
    print(f"Validation total rank error: {valid_rank_err}")


if __name__ == "__main__":
    main()