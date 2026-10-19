#!/usr/bin/env python3
"""Regenerate every result of the paper with one command.

    python run_all.py

Runs, in order,
  1. dss.py               self-test of the reference implementation,
  2. verify_examples.py   exact checks of all worked examples (Sections 4-8),
  3. simulation.py        Experiments A and B (Section 8) and Figures 1, 3, 4,
and stores each console log in results/. Exits with status 1 if any step fails.
Figure 2 (flowchart) is built separately from figures_src/fig_algorithm_flow.tex.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RES.mkdir(parents=True, exist_ok=True)

STEPS = [
    ("dss.py", "dss_selftest_output.txt"),
    ("verify_examples.py", "verify_examples_output.txt"),
    ("simulation.py", "simulation_output.txt"),
]

failed = False
for script, log in STEPS:
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE,
                          capture_output=True, text=True)
    (RES / log).write_text(proc.stdout)
    status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"{script:22s} {status:18s} {time.time() - t0:5.1f} s  -> results/{log}")
    if proc.returncode != 0:
        failed = True
        sys.stderr.write(proc.stderr)

print("\nAll results regenerated." if not failed else "\nSome steps failed.")
sys.exit(1 if failed else 0)
