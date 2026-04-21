#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export task dependency fields from perf_swimlane JSON to CSV."
    )
    parser.add_argument("input_json", type=Path, help="Path to perf_swimlane_*.json")
    parser.add_argument(
        "-o",
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: same dir as input, name task_deps_export.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_json = args.input_json
    output_csv = args.output_csv or input_json.with_name("task_deps_export.csv")

    data = json.loads(input_json.read_text())
    tasks = data.get("tasks", [])

    fieldnames = [
        "task_id",
        "ring_id",
        "func_id",
        "core_id",
        "core_type",
        "fanout_task_ids",
        "fanout_count",
        "fanin_count",
        "fanin_refcount",
    ]

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for task in tasks:
            writer.writerow(
                {
                    "task_id": task.get("task_id"),
                    "ring_id": task.get("ring_id"),
                    "func_id": task.get("func_id"),
                    "core_id": task.get("core_id"),
                    "core_type": task.get("core_type"),
                    "fanout_task_ids": ";".join(map(str, task.get("fanout", []))),
                    "fanout_count": task.get("fanout_count"),
                    "fanin_count": task.get("fanin_count"),
                    "fanin_refcount": task.get("fanin_refcount"),
                }
            )

    print(output_csv)


if __name__ == "__main__":
    main()
