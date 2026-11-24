#!/usr/bin/env bash
# Full pipeline: fetch -> tests -> popularity model -> jackpot value -> random-bet floor -> tickets -> web tool.
# bash run_all.sh --sim also reruns the f-recovery simulation (about 20 min on 8 cores).
set -euo pipefail
cd "$(dirname "$0")"
python -m pytest -q tests
python src/fetch_data.py
python src/a_randomness.py   > out/a.log
python src/d_predict.py      > out/d.log
python src/d2_hot.py         > out/d2.log
python src/b_popularity.py   > out/b.log
python src/c_jackpot.py      > out/c.log
if [[ "${1:-}" == "--sim" ]]; then python src/f_floor_sim.py > out/f_sim.log 2>&1; fi
python src/f_floor.py        > out/f.log 2>&1   # about 10 min (year bootstrap)
python src/e_generator.py
python src/build_tool.py
echo "done: results in out/*.json, tool in web/split_the_pot.html"
