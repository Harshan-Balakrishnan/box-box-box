import json
import sys

COMPOUND_PARAMS = {
    "SOFT": {
    "compound_delta": -0.96,
    "fresh_bonus": -0.35,
    "second_lap_bonus": -0.04,
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
    "compound_delta": 0.0,
    "fresh_bonus": -0.03,
    "second_lap_bonus": -0.03,
    "grace_laps": 2,
    "linear_deg": 0.016,
    "quadratic_deg": 0.001,
    "temp_deg": 0.0014,
    "late_stint_start": 999,
    "late_stint_deg": 0.0,
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


def normalize_compound(compound):
    compound = str(compound).upper()
    if compound in COMPOUND_PARAMS:
        return compound
    return "MEDIUM"


def lap_time(base_lap_time, compound, tire_age, track_temp):
    compound = normalize_compound(compound)
    params = COMPOUND_PARAMS[compound]

    effective_age = max(0, tire_age - params["grace_laps"])
    temp_offset = float(track_temp) - 30.0

    base_degradation = (
        params["linear_deg"] * effective_age
        + params["quadratic_deg"] * (effective_age ** 2)
        + params["temp_deg"] * temp_offset * effective_age
    )

    late_stint_penalty = 0.0
    if compound == "HARD":
        late_stint_age = max(0, effective_age - params["late_stint_start"])
        late_stint_penalty = params["late_stint_deg"] * (late_stint_age ** 2)

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
    base_lap_time = float(race_config["base_lap_time"])
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
        total_time += lap_time(base_lap_time, current_tire, tire_age, track_temp)

        if lap in pit_map:
            stop = pit_map[lap]
            total_time += pit_lane_time
            current_tire = normalize_compound(stop["to_tire"])
            tire_age = 0

    return total_time


def predict_finishing_positions(data):
    race_config = data["race_config"]
    results = []

    for strategy in data["strategies"].values():
        driver_id = strategy["driver_id"]
        total_time = simulate_driver(race_config, strategy)
        results.append((total_time, driver_id))

    results.sort(key=lambda item: (item[0], item[1]))

    return {
        "race_id": data["race_id"],
        "finishing_positions": [driver_id for _, driver_id in results],
    }


def main():
    data = json.load(sys.stdin)
    output = predict_finishing_positions(data)
    sys.stdout.write(json.dumps(output))


if __name__ == "__main__":
    main()