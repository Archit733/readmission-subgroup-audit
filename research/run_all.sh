#!/usr/bin/env bash
# Reproduces every number and figure. About 25 minutes on two CPU cores.
# Reads ../data/diabetic_data.csv, the copy already in this repository.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results figures
python3 scripts/01_leakage.py             | tee results/p1_log.txt
python3 scripts/02_cheap_explanations.py  | tee results/p2_log.txt
python3 scripts/03_decomposition.py all   | tee results/p3_log.txt
python3 scripts/04_interventions.py       | tee results/p4_log.txt
python3 scripts/05_figures.py
