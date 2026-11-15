"""Re-run the analysis from the released CSV logs and check that it reproduces the paper's tables.

    python experiments/verify_tables.py

Runs ``experiments/analyze_results.py`` into a temporary folder and compares every regenerated file with
``paper/generated/*.tex`` (the table bodies and number macros that the manuscript inputs) byte for byte.
Works in the repository and in the unpacked anonymized supplementary archive. Exit code 0 means every
table and statistic in the paper is reproduced from the logs.
"""

from __future__ import annotations

import filecmp
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    gen = ROOT / "paper" / "generated"
    files = sorted(gen.glob("*.tex"))
    if not files:
        print(f"no generated paper files found in {gen}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        out_gen = Path(tmp) / "gen"
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT)}
        r = subprocess.run(
            [sys.executable, str(ROOT / "experiments" / "analyze_results.py"), "--in-exp", str(ROOT / "docs" / "experiments"),
             "--out-exp", str(Path(tmp) / "exp"), "--out-gen", str(out_gen)],
            capture_output=True, text=True, env=env,
        )
        if r.returncode != 0:
            print(r.stderr[-2000:], file=sys.stderr)
            return 1
        bad = [f.name for f in files if not (out_gen / f.name).exists() or not filecmp.cmp(f, out_gen / f.name, shallow=False)]
    if bad:
        print("regenerated files differ from paper/generated: " + ", ".join(bad), file=sys.stderr)
        return 1
    print(f"OK: the analysis re-run from docs/experiments reproduces all {len(files)} generated paper files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
