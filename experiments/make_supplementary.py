"""Build the anonymized TMLR supplementary ZIP (code + logs + analysis), max 100 MB.

    python experiments/make_supplementary.py            # -> dist/supplementary_anonymized.zip

Included: README.md, ENVIRONMENT.md, pyproject.toml, requirements-lock.txt, src/, tests/, experiments/
(scripts), docs/experiments/ (CSV logs, figures, findings, pre-registration), and paper/generated/ (the table
bodies and number macros the manuscript inputs, so experiments/verify_tables.py can check them). Excluded:
.git, data/, the rest of paper/ (manuscript source carries the author block), handoff.md and review notes,
third-party mirrors (docs/references/), caches.

Identifying strings are collected at build time (git user.name / user.email, every commit author, and the
\\author{...} block of paper/main.tex) and the build fails if any of them occurs in an included file, so
the strings themselves never need to be written into this repository.
"""

from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "supplementary_anonymized.zip"
INCLUDE = ["README.md", "ENVIRONMENT.md", "pyproject.toml", "requirements-lock.txt", "src", "tests", "experiments",
           "docs/experiments", "paper/generated"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
SKIP_SUFFIX = {".pyc", ".pyo"}
MAX_BYTES = 100 * 1024 * 1024


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout
    except OSError:
        return ""


def forbidden_strings() -> list[str]:
    found: set[str] = set()
    for key in ("user.name", "user.email"):
        found.update(x.strip() for x in _git("config", "--get", key).splitlines())
    for line in _git("log", "--format=%an%n%ae").splitlines():
        found.add(line.strip())
    tex = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    m = re.search(r"\\author\{(.*?)\n\s*\n", tex, flags=re.S)
    if m:
        block = m.group(1)
        found.update(re.findall(r"\\name\s+([^\\]+?)\s*\\email", block))
        found.update(re.findall(r"\\email\s+(\S+)", block))
        found.update(x.split("@")[0] for x in re.findall(r"\\email\s+(\S+)", block))
    found.update({"github.com/" + s for s in found if s and "@" not in s and " " not in s})
    # ignore generic tokens
    return sorted(s for s in found if s and len(s) >= 4 and s.lower() not in {"unknown"})


def iter_files():
    for item in INCLUDE:
        p = ROOT / item
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not (set(f.parts) & SKIP_PARTS) and f.suffix not in SKIP_SUFFIX:
                    yield f


def check_pdf(bad: list[str]) -> int:
    """Return the number of identifying strings found in the compiled anonymous PDF (if present)."""
    pdf = ROOT / "paper" / "main.pdf"
    if not pdf.exists():
        print("paper/main.pdf not found; skipped PDF anonymity check")
        return 0
    try:
        from pypdf import PdfReader
    except ImportError:
        print("pypdf not installed; skipped PDF anonymity check")
        return 0
    reader = PdfReader(str(pdf))
    text = "\n".join(p.extract_text() or "" for p in reader.pages).lower()
    meta = " ".join(str(v) for v in (reader.metadata or {}).values()).lower()
    n = sum(1 for s in bad if s.lower() in text or s.lower() in meta)
    print(f"PDF anonymity check: {n} identifying strings found in paper/main.pdf ({len(reader.pages)} pages)")
    return n


def self_test() -> bool:
    """Unpack the archive into a clean folder and run its own experiments/verify_tables.py there: the analysis
    is re-run from the archive's CSV logs and compared byte for byte with the archive's copy of paper/generated
    (the table bodies and number macros the manuscript inputs). Reviewers can run the same script."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(OUT) as zf:
            zf.extractall(tmp)
        base = Path(tmp) / "egmm_supplementary"
        r = subprocess.run([sys.executable, str(base / "experiments" / "verify_tables.py")], capture_output=True, text=True)
        print((r.stdout or r.stderr).strip()[-2000:])
        return r.returncode == 0


def main() -> int:
    bad = forbidden_strings()
    if check_pdf(bad):
        print("The PDF is not anonymous. Fix paper/main.tex before submitting.", file=sys.stderr)
        return 1
    files = list(iter_files())
    hits = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        low = text.lower()
        for s in bad:
            if s.lower() in low:
                hits.append((f.relative_to(ROOT), s))
    if hits:
        for f, _ in hits:
            print(f"identifying string found in {f}", file=sys.stderr)
        print("Refusing to build a non-anonymous supplementary archive.", file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    import hashlib

    manifest = []
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        manifest.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {rel}")
    sys.path.insert(0, str(ROOT / "experiments"))
    from code_fingerprint import fingerprint_from_files, working_tree

    fp, _ = fingerprint_from_files(working_tree())
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, Path("egmm_supplementary") / f.relative_to(ROOT))
        zf.writestr("egmm_supplementary/MANIFEST.sha256", "\n".join(manifest) + "\n")
        zf.writestr("egmm_supplementary/LIBRARY_FINGERPRINT.txt", f"src/egmm content fingerprint (experiments/code_fingerprint.py): {fp}\n")
    if not self_test():
        print("Self-test failed: the archive does not regenerate the paper's numbers.", file=sys.stderr)
        return 1
    size = OUT.stat().st_size
    print(f"wrote {OUT.relative_to(ROOT)} ({len(files)} files, {size / 2**20:.1f} MB; checked {len(bad)} identifying strings)")
    if size > MAX_BYTES:
        print("WARNING: archive exceeds the 100 MB TMLR limit", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
