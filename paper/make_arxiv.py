"""Build the arXiv preprint (non-anonymous; run: python paper/make_arxiv.py) from paper/main.tex without touching the TMLR submission source.

Output: dist/arxiv/egmm_arxiv_source.zip (main.tex, main.bbl, tmlr.sty, tmlr.bst, references.bib, figures/, generated/)
        dist/arxiv/egmm_preprint.pdf
"""
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "dist" / "arxiv"
BUILD = OUT / "build"
TECTONIC = os.environ.get("TECTONIC", "tectonic")  # path to the tectonic binary

if OUT.exists():
    shutil.rmtree(OUT)
(BUILD / "figures").mkdir(parents=True)
(BUILD / "generated").mkdir()

src = (PAPER / "main.tex").read_text(encoding="utf-8")
anon = "\\usepackage{tmlr}               % anonymous submission (double-blind review)"
assert src.count(anon) == 1
src = src.replace(anon, "\\usepackage[preprint]{tmlr}     % de-anonymized preprint (arXiv)")
(BUILD / "main.tex").write_text(src, encoding="utf-8")
for f in ["tmlr.sty", "tmlr.bst", "references.bib"]:
    shutil.copy(PAPER / f, BUILD / f)
used = sorted({line.split("{")[-1].split("}")[0] for line in src.splitlines() if "\\includegraphics" in line})
for f in used:
    shutil.copy(PAPER / "figures" / f, BUILD / "figures" / f)
for f in sorted((PAPER / "generated").glob("*.tex")):
    shutil.copy(f, BUILD / "generated" / f.name)

r = subprocess.run([TECTONIC, "-X", "compile", "--keep-intermediates", "main.tex"], cwd=BUILD, capture_output=True, text=True)
log = r.stdout + r.stderr
bad = [l for l in log.splitlines() if ("error" in l.lower() or "overfull" in l.lower() or "undefined" in l.lower()) and "Fontconfig" not in l]
print("compile rc", r.returncode, "| problems:", bad[:5])
assert r.returncode == 0 and (BUILD / "main.bbl").exists()
shutil.copy(BUILD / "main.pdf", OUT / "egmm_preprint.pdf")

files = ["main.tex", "main.bbl", "tmlr.sty", "tmlr.bst", "references.bib"] + [f"figures/{f}" for f in used] + [f"generated/{f.name}" for f in sorted((BUILD / "generated").glob("*.tex"))]
with zipfile.ZipFile(OUT / "egmm_arxiv_source.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for f in files:
        zf.write(BUILD / f, f)

# test: unpack into a clean folder and compile from the archive alone
test = OUT / "test_unpacked"
with zipfile.ZipFile(OUT / "egmm_arxiv_source.zip") as zf:
    zf.extractall(test)
r2 = subprocess.run([TECTONIC, "-X", "compile", "main.tex"], cwd=test, capture_output=True, text=True)
print("test compile from zip rc", r2.returncode)
shutil.rmtree(test)
shutil.rmtree(BUILD)
print("files in zip:", len(files), "| zip size %.2f MB" % ((OUT / "egmm_arxiv_source.zip").stat().st_size / 2**20))
