.PHONY: setup test pipeline pdf clean

setup:            ## create venv and install (CPU)
	uv sync --extra dev

setup-nvidia:     ## install with NVIDIA RAPIDS / cuGraph-PyG extras (CUDA 12)
	uv sync --extra dev && uv pip install -r requirements-nvidia.txt --extra-index-url https://pypi.nvidia.com

test:
	uv run pytest -q

pipeline:         ## end-to-end: twin -> simulate -> baselines -> GNN -> intent -> closed loop
	uv run gnn-healing pipeline --episodes 120 --epochs 30 --out results

pdf:              ## render results into the monograph and README, then compile
	python3 scripts/results_to_tex.py && python3 scripts/results_to_readme.py
	cd docs/math && latexmk -pdf -interaction=nonstopmode main.tex > /dev/null && cp main.pdf geometric-network-healing.pdf

diagrams:         ## re-render the TikZ architecture diagrams to PDF + PNG
	cd docs/design && for f in architecture closed_loop; do pdflatex -interaction=nonstopmode $$f.tex > /dev/null && pdftoppm -png -r 170 -singlefile $$f.pdf $$f; done; rm -f docs/design/*.aux docs/design/*.log

clean:
	cd docs/math && latexmk -C; rm -rf results/*.json results/*.md
