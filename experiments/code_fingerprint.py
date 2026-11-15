"""Content fingerprint of the library code (src/egmm/*.py), independent of git and of file timestamps.

The fingerprint is SHA-256 over the sorted lines "<relative path> <sha256 of file bytes with LF line endings>".
The paper cites the fingerprint of the code frozen for the pre-registered holdout, so that anyone holding
the supplementary material can check that the library they run is the frozen one.

    python experiments/code_fingerprint.py                 # fingerprint of the working tree
    python experiments/code_fingerprint.py --git <commit>  # fingerprint of src/egmm at a commit
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _norm(b: bytes) -> bytes:
    return b.replace(b"\r\n", b"\n")


def fingerprint_from_files(files: dict[str, bytes]) -> tuple[str, list[str]]:
    lines = [f"{path} {hashlib.sha256(_norm(data)).hexdigest()}" for path, data in sorted(files.items())]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest(), lines


def working_tree() -> dict[str, bytes]:
    return {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted((ROOT / "src" / "egmm").glob("*.py"))}


def at_commit(commit: str) -> dict[str, bytes]:
    names = subprocess.run(["git", "ls-tree", "--name-only", f"{commit}:src/egmm"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.split()
    out = {}
    for n in names:
        if n.endswith(".py"):
            out[f"src/egmm/{n}"] = subprocess.run(["git", "show", f"{commit}:src/egmm/{n}"], cwd=ROOT,
                                                  capture_output=True, check=True).stdout
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--git", default=None)
    a = ap.parse_args()
    fp, lines = fingerprint_from_files(at_commit(a.git) if a.git else working_tree())
    print("\n".join(lines))
    print(f"FINGERPRINT {fp}")
