#!/usr/bin/env python3
"""Generate official-aligned CharacterEval summary CSV for deepseek-v2-lite-16b."""

from pathlib import Path
import sys


CHARACTEREVAL_ROOT = Path(__file__).resolve().parents[1]
if str(CHARACTEREVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(CHARACTEREVAL_ROOT))

from summary_utils import write_summary_csv


results_dir = CHARACTEREVAL_ROOT / "deepseek-v2-lite-16b" / "results"
output_file = CHARACTEREVAL_ROOT / "deepseek-v2-lite-16b" / "charactereval_results_summary.csv"

rows = write_summary_csv(results_dir, output_file)
print(f"Saved {len(rows)} rows to {output_file}")
