"""
Discrete-event packaging line simulation using SimPy.
Exports OEE-style KPIs for Streamlit dashboard.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import simpy


class PackagingLine:
    def __init__(
        self,
        env: simpy.Environment,
        cycle_time_sec: float = 0.8,
        defect_probability: float = 0.07,
        downtime_probability: float = 0.01,
    ) -> None:
        self.env = env
        self.cycle_time_sec = cycle_time_sec
        self.defect_probability = defect_probability
        self.downtime_probability = downtime_probability

        self.total = 0
        self.good = 0
        self.reject = 0
        self.downtime_sec = 0.0

    def run(self):
        while True:
            if random.random() < self.downtime_probability:
                stop_time = random.uniform(12, 30)
                self.downtime_sec += stop_time
                yield self.env.timeout(stop_time)

            yield self.env.timeout(self.cycle_time_sec)
            self.total += 1
            if random.random() < self.defect_probability:
                self.reject += 1
            else:
                self.good += 1


def compute_oee(total_time: float, line: PackagingLine) -> dict:
    runtime = max(1e-6, total_time - line.downtime_sec)
    ideal_output = runtime / line.cycle_time_sec
    availability = runtime / max(1e-6, total_time)
    performance = line.total / max(1e-6, ideal_output)
    quality = line.good / max(1, line.total)
    oee = availability * performance * quality

    throughput_per_hour = round((line.total / max(1e-6, total_time)) * 3600)
    defect_rate = round((line.reject / max(1, line.total)) * 100, 2)

    return {
        "simulated_seconds": round(total_time, 2),
        "total_count": line.total,
        "good_count": line.good,
        "reject_count": line.reject,
        "throughput_per_hour": throughput_per_hour,
        "defect_rate": defect_rate,
        "availability_percent": round(availability * 100, 2),
        "performance_percent": round(performance * 100, 2),
        "quality_percent": round(quality * 100, 2),
        "oee_percent": round(oee * 100, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SimPy packaging line + OEE calculator")
    parser.add_argument("--duration-sec", type=int, default=900, help="Simulation duration in seconds")
    parser.add_argument("--cycle-time-sec", type=float, default=0.8, help="Ideal pack cycle time")
    parser.add_argument("--defect-prob", type=float, default=0.07, help="Probability of package defect")
    parser.add_argument("--downtime-prob", type=float, default=0.01, help="Probability of downtime event")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("simpy_oee_metrics.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    random.seed(42)
    env = simpy.Environment()
    line = PackagingLine(
        env=env,
        cycle_time_sec=args.cycle_time_sec,
        defect_probability=args.defect_prob,
        downtime_probability=args.downtime_prob,
    )
    env.process(line.run())
    env.run(until=args.duration_sec)

    metrics = compute_oee(total_time=args.duration_sec, line=line)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved OEE metrics to: {args.output}")


if __name__ == "__main__":
    main()
