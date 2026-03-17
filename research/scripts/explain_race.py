import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import solution.race_simulator as race_simulator


def normalize_compound(compound: str) -> str:
    return race_simulator.normalize_compound(compound)


def build_stints(total_laps: int, strategy: dict):
    stints = []
    current_tire = normalize_compound(strategy["starting_tire"])
    prev_lap = 0

    for stop in strategy.get("pit_stops", []):
        stop_lap = int(stop["lap"])
        stints.append(
            {
                "compound": current_tire,
                "start_lap": prev_lap + 1,
                "end_lap": stop_lap,
                "length": stop_lap - prev_lap,
            }
        )
        current_tire = normalize_compound(stop["to_tire"])
        prev_lap = stop_lap

    stints.append(
        {
            "compound": current_tire,
            "start_lap": prev_lap + 1,
            "end_lap": total_laps,
            "length": total_laps - prev_lap,
        }
    )
    return stints


def explain_lap(base_lap_time: float, compound: str, tire_age: int, track_temp: float):
    compound = normalize_compound(compound)
    params = race_simulator.COMPOUND_PARAMS[compound]

    effective_age = max(0, tire_age - params["grace_laps"])
    temp_offset = float(track_temp) - 30.0

    base_degradation = (
        params["linear_deg"] * effective_age
        + params["quadratic_deg"] * (effective_age ** 2)
        + params["temp_deg"] * temp_offset * effective_age
    )

    late_stint_penalty = 0.0
    late_stint_age = 0
    if compound in ("HARD", "MEDIUM"):
        late_stint_age = max(0, effective_age - params["late_stint_start"])
        late_stint_penalty = params["late_stint_deg"] * (late_stint_age ** 2)

        if compound == "HARD" and tire_age >= race_simulator.HARD_ULTRA_LONG_RELIEF_START:
            ultra_long_laps = tire_age - race_simulator.HARD_ULTRA_LONG_RELIEF_START + 1
            late_stint_penalty += race_simulator.HARD_ULTRA_LONG_RELIEF_PER_LAP * ultra_long_laps

    fresh_bonus = 0.0
    if tire_age == 1:
        fresh_bonus = params["fresh_bonus"]
    elif tire_age == 2:
        fresh_bonus = params["second_lap_bonus"]

    soft_stint_adjustment = 0.0
    sprint_bonus_applied = 0.0
    fade_penalty_applied = 0.0

    if compound == "SOFT":
        if params["sprint_window_start"] <= tire_age <= params["sprint_window_end"]:
            sprint_bonus_applied = params["sprint_window_bonus"]
            soft_stint_adjustment += sprint_bonus_applied

        fade_laps = max(0, tire_age - params["fade_start"] + 1)
        fade_penalty_applied = params["fade_per_lap"] * fade_laps
        soft_stint_adjustment += fade_penalty_applied

    total = (
        float(base_lap_time)
        + params["compound_delta"]
        + fresh_bonus
        + base_degradation
        + late_stint_penalty
        + soft_stint_adjustment
    )

    return {
        "compound": compound,
        "tire_age": tire_age,
        "effective_age": effective_age,
        "base_lap_time": float(base_lap_time),
        "compound_delta": params["compound_delta"],
        "fresh_bonus": fresh_bonus,
        "base_degradation": base_degradation,
        "late_stint_age": late_stint_age,
        "late_stint_penalty": late_stint_penalty,
        "sprint_bonus_applied": sprint_bonus_applied,
        "fade_penalty_applied": fade_penalty_applied,
        "soft_stint_adjustment": soft_stint_adjustment,
        "lap_time": total,
    }


def simulate_driver_explained(race_config: dict, strategy: dict):
    total_laps = int(race_config["total_laps"])
    base_lap_time = float(race_config["base_lap_time"])
    pit_lane_time = float(race_config["pit_lane_time"])
    track_temp = float(race_config["track_temp"])

    current_tire = normalize_compound(strategy["starting_tire"])
    tire_age = 0
    total_time = 0.0

    pit_map = {}
    for stop in strategy.get("pit_stops", []):
        pit_map[int(stop["lap"])] = stop

    lap_rows = []
    stint_totals = []
    current_stint_total = 0.0
    current_stint_laps = []

    for lap in range(1, total_laps + 1):
        tire_age += 1
        detail = explain_lap(base_lap_time, current_tire, tire_age, track_temp)
        total_time += detail["lap_time"]
        current_stint_total += detail["lap_time"]
        current_stint_laps.append((lap, copy.deepcopy(detail)))

        lap_rows.append(
            {
                "lap": lap,
                "compound": current_tire,
                "tire_age": tire_age,
                **detail,
            }
        )

        if lap in pit_map:
            stop = pit_map[lap]
            total_time += pit_lane_time
            stint_totals.append(
                {
                    "compound": current_tire,
                    "end_lap": lap,
                    "lap_time_sum": current_stint_total,
                    "pit_penalty_after_stint": pit_lane_time,
                    "stint_total_with_pit": current_stint_total + pit_lane_time,
                    "laps": current_stint_laps,
                }
            )
            current_tire = normalize_compound(stop["to_tire"])
            tire_age = 0
            current_stint_total = 0.0
            current_stint_laps = []

    if current_stint_laps:
        stint_totals.append(
            {
                "compound": current_tire,
                "end_lap": total_laps,
                "lap_time_sum": current_stint_total,
                "pit_penalty_after_stint": 0.0,
                "stint_total_with_pit": current_stint_total,
                "laps": current_stint_laps,
            }
        )

    return {
        "total_time": total_time,
        "lap_rows": lap_rows,
        "stint_totals": stint_totals,
    }


def predict_with_details(data: dict):
    race_config = data["race_config"]
    results = []

    for pos_key, strategy in data["strategies"].items():
        driver_id = strategy["driver_id"]
        grid_position = int(str(pos_key).replace("pos", ""))
        sim = simulate_driver_explained(race_config, strategy)
        results.append(
            {
                "driver_id": driver_id,
                "grid_position": grid_position,
                "strategy": strategy,
                "total_time": sim["total_time"],
                "lap_rows": sim["lap_rows"],
                "stint_totals": sim["stint_totals"],
            }
        )

    results.sort(key=lambda row: (row["total_time"], row["grid_position"], row["driver_id"]))

    predicted_order = [row["driver_id"] for row in results]
    actual_order = data.get("finishing_positions")

    actual_rank = {}
    if actual_order:
        actual_rank = {driver_id: i + 1 for i, driver_id in enumerate(actual_order)}

    for i, row in enumerate(results, start=1):
        row["predicted_rank"] = i
        row["actual_rank"] = actual_rank.get(row["driver_id"])
        if row["actual_rank"] is not None:
            row["rank_error"] = row["predicted_rank"] - row["actual_rank"]
        else:
            row["rank_error"] = None

    return results, predicted_order, actual_order


def print_driver_summary(row: dict, total_laps: int):
    strategy = row["strategy"]
    stints = build_stints(total_laps, strategy)

    print("=" * 80)
    print(
        f"Driver {row['driver_id']} | grid={row['grid_position']} | "
        f"pred={row['predicted_rank']}"
        + (
            f" | actual={row['actual_rank']} | rank_error={row['rank_error']:+d}"
            if row["actual_rank"] is not None
            else ""
        )
    )
    print(f"Total simulated time: {row['total_time']:.6f}")
    print(f"Starting tire: {normalize_compound(strategy['starting_tire'])}")

    if strategy.get("pit_stops"):
        print("Pit stops:")
        for stop in strategy["pit_stops"]:
            print(f"  lap {int(stop['lap'])} -> {normalize_compound(stop['to_tire'])}")
    else:
        print("Pit stops: none")

    print("Stints:")
    for stint in stints:
        print(
            f"  {stint['compound']}: laps {stint['start_lap']}-{stint['end_lap']} "
            f"(len={stint['length']})"
        )

    print("Stint contributions:")
    for idx, stint in enumerate(row["stint_totals"], start=1):
        start_lap = stint["laps"][0][0]
        end_lap = stint["laps"][-1][0]
        print(
            f"  Stint {idx}: {stint['compound']} laps {start_lap}-{end_lap} | "
            f"lap_sum={stint['lap_time_sum']:.6f} | "
            f"pit_after={stint['pit_penalty_after_stint']:.6f} | "
            f"stint_total={stint['stint_total_with_pit']:.6f}"
        )

    print("Lap details:")
    for lap, detail in [(r["lap"], r) for r in row["lap_rows"]]:
        print(
            f"  lap {lap:>2} | {detail['compound']:<6} age={detail['tire_age']:>2} "
            f"| delta={detail['compound_delta']:+.4f} "
            f"| fresh={detail['fresh_bonus']:+.4f} "
            f"| deg={detail['base_degradation']:+.4f} "
            f"| late={detail['late_stint_penalty']:+.4f} "
            f"| soft_adj={detail['soft_stint_adjustment']:+.4f} "
            f"| lap_time={detail['lap_time']:.6f}"
        )


def print_pairwise_gaps(results):
    print("\n" + "#" * 80)
    print("Predicted order gaps")
    print("#" * 80)
    for i in range(len(results) - 1):
        a = results[i]
        b = results[i + 1]
        gap = b["total_time"] - a["total_time"]
        print(
            f"{i+1:>2}. {a['driver_id']} -> {b['driver_id']} | "
            f"time_gap={gap:.6f} | "
            f"grid {a['grid_position']} vs {b['grid_position']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Explain one race prediction in detail.")
    parser.add_argument("race_json", help="Path to race json file")
    parser.add_argument(
        "--driver",
        help="Optional driver id to print only one driver explanation",
        default=None,
    )
    args = parser.parse_args()

    race_path = Path(args.race_json)
    with race_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results, predicted_order, actual_order = predict_with_details(data)
    total_laps = int(data["race_config"]["total_laps"])

    print("#" * 80)
    print(f"Race: {data.get('race_id', '<unknown>')}")
    print(
        f"Config: total_laps={data['race_config']['total_laps']}, "
        f"base_lap_time={data['race_config']['base_lap_time']}, "
        f"pit_lane_time={data['race_config']['pit_lane_time']}, "
        f"track_temp={data['race_config']['track_temp']}"
    )
    print("#" * 80)

    print("\nPredicted order:")
    print(predicted_order)

    if actual_order:
        print("\nActual order:")
        print(actual_order)

    if args.driver:
        matched = [row for row in results if row["driver_id"] == args.driver]
        if not matched:
            print(f"\nDriver {args.driver} not found.")
            sys.exit(1)
        print()
        print_driver_summary(matched[0], total_laps)
    else:
        for row in results:
            print()
            print_driver_summary(row, total_laps)

    print_pairwise_gaps(results)


if __name__ == "__main__":
    main()