import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

from solution.race_simulator import predict_finishing_positions


def load_historical_races(historical_dir: Path, limit: int | None = None):
    files = sorted(historical_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No historical race files found in {historical_dir}")

    races = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            chunk = json.load(handle)

        if not isinstance(chunk, list):
            raise ValueError(f"Expected a list of races in {path}")

        races.extend(chunk)

        if limit is not None and len(races) >= limit:
            return races[:limit]

    return races


def build_pattern(strategy: dict, total_laps: int) -> str:
    stints = []
    current_tire = str(strategy["starting_tire"]).upper()
    last_lap = 0

    for stop in strategy.get("pit_stops", []):
        lap = int(stop["lap"])
        stints.append(f"{current_tire}:{lap - last_lap}")
        current_tire = str(stop["to_tire"]).upper()
        last_lap = lap

    stints.append(f"{current_tire}:{total_laps - last_lap}")
    return " -> ".join(stints)


def evaluate_races(races: list[dict]):
    exact_matches = 0

    pattern_stats = defaultdict(
        lambda: {
            "samples": 0,
            "wrong_samples": 0,
            "total_abs_rank_error": 0,
            "too_high": 0,
            "too_low": 0,
            "example_race_id": None,
            "example_driver_id": None,
            "example_signed_error": None,
        }
    )

    worst_races = []

    for race in races:
        predicted = predict_finishing_positions(copy.deepcopy(race))
        predicted_order = predicted["finishing_positions"]
        actual_order = race["finishing_positions"]

        is_exact = predicted_order == actual_order
        if is_exact:
            exact_matches += 1

        predicted_pos = {
            driver_id: index + 1
            for index, driver_id in enumerate(predicted_order)
        }
        actual_pos = {
            driver_id: index + 1
            for index, driver_id in enumerate(actual_order)
        }

        race_abs_rank_error = 0
        wrong_driver_count = 0
        total_laps = int(race["race_config"]["total_laps"])

        for strategy in race["strategies"].values():
            driver_id = strategy["driver_id"]
            signed_error = predicted_pos[driver_id] - actual_pos[driver_id]
            abs_error = abs(signed_error)
            pattern = build_pattern(strategy, total_laps)
            stats = pattern_stats[pattern]

            stats["samples"] += 1
            stats["total_abs_rank_error"] += abs_error

            if abs_error > 0:
                stats["wrong_samples"] += 1
                wrong_driver_count += 1
                race_abs_rank_error += abs_error

                if signed_error < 0:
                    stats["too_high"] += 1
                elif signed_error > 0:
                    stats["too_low"] += 1

                if stats["example_race_id"] is None:
                    stats["example_race_id"] = race["race_id"]
                    stats["example_driver_id"] = driver_id
                    stats["example_signed_error"] = signed_error

        worst_races.append(
            {
                "race_id": race["race_id"],
                "is_exact": is_exact,
                "wrong_driver_count": wrong_driver_count,
                "total_abs_rank_error": race_abs_rank_error,
            }
        )

    total_races = len(races)
    exact_match_accuracy = exact_matches / total_races if total_races else 0.0

    pattern_rows = []
    for pattern, stats in pattern_stats.items():
        avg_abs_rank_error = stats["total_abs_rank_error"] / stats["samples"]
        wrong_rate = stats["wrong_samples"] / stats["samples"]

        pattern_rows.append(
            {
                "pattern": pattern,
                "samples": stats["samples"],
                "wrong_samples": stats["wrong_samples"],
                "wrong_rate": wrong_rate,
                "avg_abs_rank_error": avg_abs_rank_error,
                "too_high": stats["too_high"],
                "too_low": stats["too_low"],
                "example_race_id": stats["example_race_id"],
                "example_driver_id": stats["example_driver_id"],
                "example_signed_error": stats["example_signed_error"],
            }
        )

    pattern_rows.sort(
        key=lambda row: (
            -row["wrong_rate"],
            -row["samples"],
            -row["avg_abs_rank_error"],
            row["pattern"],
        )
    )

    worst_races.sort(
        key=lambda row: (
            -int(not row["is_exact"]),
            -row["wrong_driver_count"],
            -row["total_abs_rank_error"],
            row["race_id"],
        )
    )

    return {
        "total_races": total_races,
        "exact_matches": exact_matches,
        "exact_match_accuracy": exact_match_accuracy,
        "pattern_rows": pattern_rows,
        "worst_races": worst_races,
    }


def print_report(summary: dict, top_patterns: int, top_races: int, min_pattern_samples: int):
    total_races = summary["total_races"]
    exact_matches = summary["exact_matches"]
    accuracy = summary["exact_match_accuracy"] * 100.0

    print("Historical evaluation")
    print("=====================")
    print(f"Total races:          {total_races}")
    print(f"Exact matches:        {exact_matches}")
    print(f"Exact-match accuracy: {accuracy:.2f}%")

    print("\nWorst races by ranking error:")
    shown_races = 0
    for row in summary["worst_races"]:
        if row["is_exact"]:
            continue
        shown_races += 1
        print(
            f"- {row['race_id']}: wrong_drivers={row['wrong_driver_count']}, "
            f"total_abs_rank_error={row['total_abs_rank_error']}"
        )
        if shown_races >= top_races:
            break
    if shown_races == 0:
        print("- None. Every race matched exactly.")

    print("\nStrategy patterns most often wrong:")
    shown_patterns = 0
    for row in summary["pattern_rows"]:
        if row["samples"] < min_pattern_samples:
            continue
        if row["wrong_samples"] == 0:
            continue

        shown_patterns += 1

        example_text = ""
        if row["example_race_id"] is not None:
            direction = "too low" if row["example_signed_error"] > 0 else "too high"
            example_text = (
                f", example={row['example_race_id']} {row['example_driver_id']} "
                f"({direction} by {abs(row['example_signed_error'])} places)"
            )

        print(
            f"- {row['pattern']}: samples={row['samples']}, "
            f"wrong={row['wrong_samples']} ({row['wrong_rate'] * 100.0:.1f}%), "
            f"avg_abs_rank_error={row['avg_abs_rank_error']:.3f}, "
            f"too_high={row['too_high']}, too_low={row['too_low']}{example_text}"
        )

        if shown_patterns >= top_patterns:
            break

    if shown_patterns == 0:
        print("- No wrong patterns met the minimum sample threshold.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the current solver on historical races."
    )
    parser.add_argument(
        "--historical-dir",
        default="data/historical_races",
        help="Folder containing historical race JSON files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of races to evaluate.",
    )
    parser.add_argument(
        "--top-patterns",
        type=int,
        default=12,
        help="How many strategy patterns to show.",
    )
    parser.add_argument(
        "--top-races",
        type=int,
        default=10,
        help="How many of the worst races to show.",
    )
    parser.add_argument(
        "--min-pattern-samples",
        type=int,
        default=20,
        help="Only show patterns with at least this many samples.",
    )
    args = parser.parse_args()

    races = load_historical_races(Path(args.historical_dir), args.limit)
    summary = evaluate_races(races)
    print_report(
        summary,
        top_patterns=args.top_patterns,
        top_races=args.top_races,
        min_pattern_samples=args.min_pattern_samples,
    )


if __name__ == "__main__":
    main()