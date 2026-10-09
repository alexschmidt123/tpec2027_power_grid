#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build output/pdf
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build eig_power_grid_conference.tex
pushd build >/dev/null
BIBINPUTS=..: bibtex eig_power_grid_conference
popd >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build eig_power_grid_conference.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build eig_power_grid_conference.tex
cp build/eig_power_grid_conference.pdf output/pdf/eig_power_grid_conference.pdf
