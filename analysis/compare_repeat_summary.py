from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = (
    PROJECT_ROOT
    / "log"
    / "swinv2_tiny"
    / "blockD_swinv2_tiny_224_3seeds"
    / "repeat_summary.json"
)
DEFAULT_CANDIDATE = (
    PROJECT_ROOT
    / "log"
    / "swinv2_tiny_fpn"
    / "blockE_swinv2_tiny_fpn_224_3seeds"
    / "repeat_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two repeat_summary.json files by best_top1_mean."
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--metric", default="best_top1_mean")
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_metric(summary: dict, metric: str, path: Path) -> float:
    if metric not in summary:
        raise KeyError(f"Missing metric '{metric}' in {path}")
    return float(summary[metric])


def main() -> int:
    args = parse_args()
    baseline = load_summary(args.baseline)
    candidate = load_summary(args.candidate)

    baseline_value = get_metric(baseline, args.metric, args.baseline)
    candidate_value = get_metric(candidate, args.metric, args.candidate)
    delta = candidate_value - baseline_value

    print(f"metric={args.metric}")
    print(f"baseline={args.baseline}")
    print(f"baseline_value={baseline_value:.4f}")
    print(f"candidate={args.candidate}")
    print(f"candidate_value={candidate_value:.4f}")
    print(f"delta={delta:+.4f}")

    if candidate_value > baseline_value:
        print("result=PASS candidate is higher than baseline")
        return 0

    print("result=FAIL candidate is not higher than baseline")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        raise SystemExit(1)
