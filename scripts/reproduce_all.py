#!/usr/bin/env python3
"""Verify cached results or run the manuscript experiment pipelines."""
from __future__ import annotations
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def commands(mode, stages):
    if mode == "verify":
        return [[sys.executable, "scripts/validate_precomputed_results.py", "--geometry"],
                [sys.executable, "scripts/regenerate_manuscript_artifacts.py"]]
    result = []
    for stage in stages:
        if stage in {"strengthening", "confirmation"}:
            name = "run_tnnls_strengthening.py" if stage == "strengthening" else "run_step5_confirmation.py"
            command = [sys.executable, f"experiments/strengthening/{name}"]
            if mode == "paper":
                command.append("--full")
        elif stage == "extended":
            command = [sys.executable, "experiments/run_extended_experiments.py", "--mode",
                       "paper"]
        else:
            command = [sys.executable, "experiments/run_principal_scalar.py", "--mode",
                       "paper" if mode == "paper" else "smoke"]
        result.append(command)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("verify", "quick", "paper"), default="verify")
    parser.add_argument("--stages", nargs="+", choices=("strengthening", "confirmation", "extended", "legacy"),
                        default=["strengthening", "confirmation", "extended"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and args.mode != "verify" and importlib.util.find_spec("torch") is None:
        parser.error("Training requires PyTorch. Install requirements.txt; verification needs only requirements-analysis.txt.")
    for command in commands(args.mode, args.stages):
        print("$", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
