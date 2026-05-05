#!/usr/bin/env python3
"""Generate official-aligned CharacterEval summary CSV for llama70B."""

from pathlib import Path
import sys


CHARACTEREVAL_ROOT = Path(__file__).resolve().parents[1]
if str(CHARACTEREVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(CHARACTEREVAL_ROOT))

from summary_utils import write_summary_csv


results_dir = CHARACTEREVAL_ROOT / "llama70B" / "results"
output_file = CHARACTEREVAL_ROOT / "llama70B" / "charactereval_results_summary.csv"

rows = write_summary_csv(
    results_dir,
    output_file,
    exclude_models=set(),
)
print(f"Saved {len(rows)} rows to {output_file}")
