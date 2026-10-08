"""Splice results/results.md between the RESULTS markers in README.md."""
from pathlib import Path
root = Path(__file__).resolve().parents[1]
res = (root / "results" / "results.md").read_text().replace("# Results\n", "").strip()
readme = (root / "README.md").read_text()
a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
pre, rest = readme.split(a); _, post = rest.split(b)
(root / "README.md").write_text(f"{pre}{a}\n{res}\n\n*(CPU, fixed seed; regenerate with `make pipeline`.)*\n{b}{post}")
print("README results updated")
