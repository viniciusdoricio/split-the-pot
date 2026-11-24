#!/usr/bin/env bash
# Full pipeline: fetch -> tests -> popularity model -> jackpot value -> tickets -> web tool.
set -euo pipefail
cd "$(dirname "$0")"
python src/fetch_data.py
python src/a_randomness.py   > out/a.log
python src/d_predict.py      > out/d.log
python src/d2_hot.py         > out/d2.log
python src/b_popularity.py   > out/b.log
python src/c_jackpot.py      > out/c.log
python src/e_generator.py
python src/build_tool.py
echo "done: results in out/*.json, tool in web/split_the_pot.html"
