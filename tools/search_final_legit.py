import copy
import glob
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import solution.race_simulator as race_simulator


TRAIN_RACE_MAX_ID = 24000


def extract_numeric_race_id(race_id: str) -> int:
    digits = "".join(ch for ch in str(race_id) if ch.isdigit())
    return int(digits) if digits else -1


def load_all_historical_races(historical_dir: Path):
    races = []
    for path in sorted(glob.glob(str(historical_dir / "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            chunk = json.load(f)
        races.extend(chunk)
    return races


def split_train_validation(races):
    train = []
    valid = []
    for race in races:
        race_num = extract_numeric_race_id(race["race_id"])
        if race_num <= TRAIN_RACE_MAX_ID:
            train.append(race)
        else:
            valid.append(race)
    return train, valid


def deep_copy_params(params):
    return {compound: values.copy() for compound, values in params.items()}


def snapshot_state():
    return {
        "params": deep_copy_params(race_simulator.COMPOUND_PARAMS),
        "hard_ultra_start": race_simulator.HARD_ULTRA_LONG_RELIEF_START,
        "hard_ultra_per_lap": race_simulator.HARD_ULTRA_LONG_RELIEF_PER_LAP,
    }


def apply_state(state):
    race_simulator.COMPOUND_PARAMS = deep_copy_params(state["params"])
    race_simulator.HARD_ULTRA_LONG_RELIEF_START = state["hard_ultra_start"]
    race_simulator.HARD_ULTRA_LONG_RELIEF_PER_LAP = state["hard_ultra_per_lap"]


def set_value(state, key, value):
    new_state = copy.deepcopy(state)
    if key == "hard_ultra_start":
        new_state["hard_ultra_start"] = int(value)
    elif key == "hard_ultra_per_lap":
        new_state["hard_ultra_per_lap"] = float(value)
    else:
        compound, field = key.split(".", 1)
        new_state["params"][compound][field] = value
    return new_state


def get_value(state, key):
    if key == "hard_ultra_start":
        return state["hard_ultra_start"]
    if key == "hard_ultra_per_lap":
        return state["hard_ultra_per_lap"]
    compound, field = key.split(".", 1)
    return state["params"][compound][field]


def evaluate_races(races, state):
    apply_state(state)

    exact_matches = 0
    total_l1 = 0
    total_inversions = 0
    total_pairwise_correct = 0
    total_pairwise = 0

    for race in races:
        predicted = race_simulator.predict_finishing_positions(race)["finishing_positions"]
        actual = race["finishing_positions"]

        if predicted == actual:
            exact_matches += 1

        actual_pos = {driver_id: i for i, driver_id in enumerate(actual)}
        pred_pos = {driver_id: i for i, driver_id in enumerate(predicted)}

        total_l1 += sum(abs(pred_pos[d] - actual_pos[d]) for d in actual)

        n = len(actual)
        for i in range(n):
            left = predicted[i]
            for j in range(i + 1, n):
                right = predicted[j]
                total_pairwise += 1
                if actual_pos[left] < actual_pos[right]:
                    total_pairwise_correct += 1
                else:
                    total_inversions += 1

    return {
        "n_races": len(races),
        "exact_matches": exact_matches,
        "pairwise_accuracy": (
            total_pairwise_correct / total_pairwise if total_pairwise else 0.0
        ),
        "mean_l1": (total_l1 / len(races)) if races else math.inf,
        "mean_inversions": (total_inversions / len(races)) if races else math.inf,
        "total_l1": total_l1,
        "total_inversions": total_inversions,
    }


def score_tuple(summary):
    return (
        summary["exact_matches"],
        summary["pairwise_accuracy"],
        -summary["mean_l1"],
        -summary["mean_inversions"],
    )


def diff_lines(base_state, candidate_state):
    lines = []
    for compound in ("SOFT", "MEDIUM", "HARD"):
        for field, base_value in base_state["params"][compound].items():
            new_value = candidate_state["params"][compound][field]
            if new_value != base_value:
                lines.append(
                    f'COMPOUND_PARAMS["{compound}"]["{field}"] = {repr(new_value)}'
                )
    if candidate_state["hard_ultra_start"] != base_state["hard_ultra_start"]:
        lines.append(
            f"HARD_ULTRA_LONG_RELIEF_START = {candidate_state['hard_ultra_start']}"
        )
    if candidate_state["hard_ultra_per_lap"] != base_state["hard_ultra_per_lap"]:
        lines.append(
            f"HARD_ULTRA_LONG_RELIEF_PER_LAP = {candidate_state['hard_ultra_per_lap']}"
        )
    return lines


def build_search_space(base_state):
    s = base_state["params"]["SOFT"]
    m = base_state["params"]["MEDIUM"]
    h = base_state["params"]["HARD"]

    def uniq(vals):
        out = []
        for v in vals:
            if v not in out:
                out.append(v)
        return out

    return {
        "SOFT.compound_delta": uniq([
            round(s["compound_delta"] - 0.08, 4),
            round(s["compound_delta"] - 0.04, 4),
            s["compound_delta"],
            round(s["compound_delta"] + 0.04, 4),
        ]),
        "SOFT.linear_deg": uniq([
            round(s["linear_deg"] - 0.004, 5),
            round(s["linear_deg"] - 0.002, 5),
            s["linear_deg"],
            round(s["linear_deg"] + 0.002, 5),
        ]),
        "SOFT.fade_start": uniq([
            max(3, int(s["fade_start"]) - 1),
            int(s["fade_start"]),
            int(s["fade_start"]) + 1,
        ]),
        "SOFT.fade_per_lap": uniq([
            round(s["fade_per_lap"] - 0.03, 4),
            round(s["fade_per_lap"] - 0.01, 4),
            s["fade_per_lap"],
            round(s["fade_per_lap"] + 0.01, 4),
        ]),
        "MEDIUM.compound_delta": uniq([
            round(m["compound_delta"] - 0.05, 4),
            round(m["compound_delta"] - 0.02, 4),
            m["compound_delta"],
            round(m["compound_delta"] + 0.02, 4),
        ]),
        "MEDIUM.late_stint_deg": uniq([
            round(max(0.0, m["late_stint_deg"] - 0.004), 5),
            round(max(0.0, m["late_stint_deg"] - 0.002), 5),
            m["late_stint_deg"],
            round(m["late_stint_deg"] + 0.002, 5),
        ]),
        "HARD.compound_delta": uniq([
            round(h["compound_delta"] - 0.05, 4),
            round(h["compound_delta"] - 0.02, 4),
            h["compound_delta"],
            round(h["compound_delta"] + 0.02, 4),
        ]),
        "HARD.late_stint_deg": uniq([
            round(max(0.0, h["late_stint_deg"] - 0.0002), 6),
            h["late_stint_deg"],
            round(h["late_stint_deg"] + 0.0001, 6),
            round(h["late_stint_deg"] + 0.0002, 6),
        ]),
        "hard_ultra_start": uniq([
            max(10, base_state["hard_ultra_start"] - 2),
            max(10, base_state["hard_ultra_start"] - 1),
            base_state["hard_ultra_start"],
            base_state["hard_ultra_start"] + 1,
        ]),
        "hard_ultra_per_lap": uniq([
            round(base_state["hard_ultra_per_lap"] - 0.01, 4),
            round(base_state["hard_ultra_per_lap"] - 0.005, 4),
            base_state["hard_ultra_per_lap"],
            round(base_state["hard_ultra_per_lap"] + 0.005, 4),
        ]),
    }


def main():
    historical_dir = ROOT / "data" / "historical_races"
    all_races = load_all_historical_races(historical_dir)
    train_races, valid_races = split_train_validation(all_races)

    base_state = snapshot_state()
    search_space = build_search_space(base_state)

    print("Loaded historical races")
    print("=======================")
    print(f"Total:      {len(all_races)}")
    print(f"Train:      {len(train_races)}")
    print(f"Validation: {len(valid_races)}")
    print()

    base_train = evaluate_races(train_races, base_state)
    base_valid = evaluate_races(valid_races, base_state)

    print("Baseline")
    print("========")
    print(
        f"Train: exact={base_train['exact_matches']}/{base_train['n_races']}, "
        f"pairwise={base_train['pairwise_accuracy']:.4f}, "
        f"mean_l1={base_train['mean_l1']:.3f}, "
        f"mean_inv={base_train['mean_inversions']:.3f}"
    )
    print(
        f"Valid: exact={base_valid['exact_matches']}/{base_valid['n_races']}, "
        f"pairwise={base_valid['pairwise_accuracy']:.4f}, "
        f"mean_l1={base_valid['mean_l1']:.3f}, "
        f"mean_inv={base_valid['mean_inversions']:.3f}"
    )
    print()

    best_state = copy.deepcopy(base_state)
    best_train = base_train
    best_valid = base_valid

    for round_index in range(1, 4):
        improved = False
        print(f"Round {round_index}")
        print("-------")

        for key, values in search_space.items():
            current_value = get_value(best_state, key)

            local_best_state = best_state
            local_best_train = best_train
            local_best_valid = best_valid

            for candidate_value in values:
                candidate_state = set_value(best_state, key, candidate_value)
                train_summary = evaluate_races(train_races, candidate_state)
                valid_summary = evaluate_races(valid_races, candidate_state)

                candidate_score = score_tuple(valid_summary)
                local_score = score_tuple(local_best_valid)

                if candidate_score > local_score:
                    local_best_state = candidate_state
                    local_best_train = train_summary
                    local_best_valid = valid_summary

            new_value = get_value(local_best_state, key)
            if new_value != current_value:
                best_state = local_best_state
                best_train = local_best_train
                best_valid = local_best_valid
                improved = True
                print(
                    f"improved {key}: {current_value} -> {new_value} | "
                    f"valid exact={best_valid['exact_matches']}/{best_valid['n_races']}, "
                    f"pairwise={best_valid['pairwise_accuracy']:.4f}, "
                    f"mean_l1={best_valid['mean_l1']:.3f}, "
                    f"mean_inv={best_valid['mean_inversions']:.3f}"
                )

        if not improved:
            print("no further improvement found")
            print()
            break
        print()

    print("Best validation result")
    print("======================")
    print(
        f"Train: exact={best_train['exact_matches']}/{best_train['n_races']}, "
        f"pairwise={best_train['pairwise_accuracy']:.4f}, "
        f"mean_l1={best_train['mean_l1']:.3f}, "
        f"mean_inv={best_train['mean_inversions']:.3f}"
    )
    print(
        f"Valid: exact={best_valid['exact_matches']}/{best_valid['n_races']}, "
        f"pairwise={best_valid['pairwise_accuracy']:.4f}, "
        f"mean_l1={best_valid['mean_l1']:.3f}, "
        f"mean_inv={best_valid['mean_inversions']:.3f}"
    )
    print()

    print("Patch lines")
    print("===========")
    for line in diff_lines(base_state, best_state):
        print(line)


if __name__ == "__main__":
    main()