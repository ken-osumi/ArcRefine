"""Command-line interface; validation does not load models or contact an MSA server."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from structural_carryover.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON input; paths are relative to this file")
    parser.add_argument("--output", type=Path, help="new output directory (required for a run)")
    parser.add_argument(
        "--validate-only", action="store_true", help="validate inputs without a GPU"
    )
    parser.add_argument("--carryover", choices=["on", "off"], help="override outer-loop carryover")
    parser.add_argument("--steps", type=int, nargs=3, metavar=("SOFT", "SHARPEN", "FINAL"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-predict", action="store_true", help="skip the independent parent and final predictions")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.carryover is not None:
            config = replace(config, carryover=args.carryover == "on")
        if args.steps is not None:
            if any(s < 1 for s in args.steps):
                raise ValueError("all three phase lengths must be positive")
            config = replace(config, phase_steps=tuple(args.steps))
        if args.seed is not None:
            if not 0 <= args.seed < 2**32:
                raise ValueError("seed must be in [0, 2**32)")
            config = replace(config, seed=args.seed)
        from structural_carryover.inputs import validate_target_files

        validate_target_files(config)
        if args.validate_only:
            print(json.dumps(config.to_dict(), indent=2))
            return
        if args.output is None:
            parser.error("--output is required for a run")
        from structural_carryover.runner import run

        result = run(config, args.output, predict=not args.no_predict)
        print(json.dumps(result, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"Input error: {exc}\n")


if __name__ == "__main__":
    main()
