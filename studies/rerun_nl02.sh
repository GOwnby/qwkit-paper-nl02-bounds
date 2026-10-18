#!/bin/sh
# Regenerate every number and figure of the NL-02 manuscript (paper_nl02/). Requires the N1 study outputs
# (studies/n1_paper/*.json; run studies/rerun_n1.sh first if they are missing). Total ~1.5 h on 2 cores.
set -e
cd "$(dirname "$0")/nl02"
python3 qwkit_validation.py   > qwkit_validation.log   # full qwkit vs mapped ladder, identity scans
python3 study_universal.py    > universal.log          # Theorems 1-2 checks, saturation curves, maps
python3 detuning_study.py     > detuning.log           # detuning rule
python3 bottleneck.py         > bottleneck.log         # triple-well bottleneck variants
python3 hw_scan.py            > hw_scan.log            # pump photon-energy scan, waveguide recomputed per wavelength
python3 asym_check.py         > asym_check.log         # large-intensity asymptote
python3 dm_check.py           > dm_check.log           # full three-level density matrix vs rate equations
python3 dm_local.py           > dm_local.log           # density matrix with locally compensated index
python3 length_bound.py       > length_bound.log       # generalized length-limited bound
python3 vurg_check.py         > vurg_check.log         # Vurgaftman 64.5/18/36.5 A structure (Relation to prior estimates)
python3 ay_check.py           > ay_check.log           # Almogy-Yariv closed-form saturated susceptibility
python3 collect_numbers.py    > numbers.log
python3 paper_figures.py
cd ../../paper_nl02 && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null && \
  pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null
cp main.pdf ../NL02_paper.pdf
