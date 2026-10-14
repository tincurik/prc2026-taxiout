#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Build main.pdf: redraw the figures, then pdflatex + biber + pdflatex twice (the JOAS Makefile's steps).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
"$PY" make_figures.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
biber --quiet main
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
rm -f main.aux main.bbl main.blg main.bcf main.log main.out main.run.xml main.synctex.gz
echo "built main.pdf ($(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/ {print $2}') pages)"
