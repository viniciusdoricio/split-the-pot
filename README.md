# Split the Pot

Which Mega-Sena numbers do people actually pick? This repo infers it from 30 years of quadra and quina winner counts (Poisson GLM), checks the draws for fairness (Monte Carlo), backtests the usual prediction strategies (walk-forward) and builds a ticket tool for the Mega da Virada 2026.

Data: all 3,089 draws of Brazil's Mega-Sena, 1996 to 2026.

Summary: nothing here changes the 1 in 50,063,860 odds. Popularity only changes how many people you'd share a jackpot with, roughly half as many on an unpopular ticket. At the Virada that's an expected ~R$96 mi if you win, vs ~R$180-250 mi. Expected value per R$6 ticket is still negative.

## Quick start
```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash run_all.sh
```
Then open `web/split_the_pot.html` in a browser.

## Results so far (to contest 3089, 22 Nov 2026)
| Question | Answer | File |
|---|---|---|
| Is the draw fair today? | Yes. 2023-2026 frequency test p = 0.64; all-time unevenness (p ≈ 0.004) comes from old eras | out/a_results.json |
| Do hot/cold/overdue/ML predict? | No out-of-sample edge in recent eras (ML AUC 0.500) | out/d_results.json, out/d2_results.json |
| Which numbers are crowded? | 13, 5, 7, 3, 10 and most of 1-31 (+8% on average); 40, 50, 41, 30 and 52-60 are the emptiest | out/popularity.csv |
| Is that real? | Stable 1998-2019 vs 2020-2026 (r = 0.93); predicts jackpot winner counts (exponent 1.63 ± 0.14 vs theory 1.5) | out/b_results.json, out/c_results.json |
| What's a ticket worth at the Virada? | Average R$2.14 per R$6; least-crowded R$3.90-5.35 depending on assumptions | out/c_results.json |

## Layout
```
src/        pipeline scripts (see Scripts below)
out/        generated results (json/csv/pkl), recreated by run_all.sh
data/       draws.csv, flat file for Excel / Power BI
web/        tool_template.html + the built split_the_pot.html
ROADMAP.md  next analyses worth doing
PLAYTEST.md the 10-31 Dec plan and pre-registered model test
```

## Scripts
- `src/fetch_data.py`: downloads all draws (winners per tier and sales) and cross-checks the numbers against a second scrape. Writes `out/draws.pkl` and `data/draws.csv`.
- `src/a_randomness.py`: Monte Carlo fairness tests.
- `src/d_predict.py`, `src/d2_hot.py`: walk-forward strategy backtests and ML.
- `src/b_popularity.py`: number popularity GLM and pattern effects. Writes `out/popularity.csv`.
- `src/c_jackpot.py`: jackpot validation and Virada EV scenarios.
- `src/e_generator.py`: low-crowd tickets and the model export for the web tool.
- `src/build_tool.py`: writes `web/split_the_pot.html` (self-contained, open in a browser).

Sales exist from 2009 on (plus a few 1998 draws). Bet prices before 2015 are approximate and only used to turn sales into bet counts.

## Sources
- Draws, winners and sales (scraped from Caixa): github.com/GabrielRossiSoares/megasena; cross-checked with github.com/sosthenis/megasena-data, github.com/guilhermeasn/loteria.json, github.com/mneves75/megasena-analyser-webapp (0 mismatches).
- Method references: Farrell & Walker on the UK lottery (inferring popularity from winner counts); Number preferences in lotteries, JDM 2016 (Dutch Lotto); JDM 2021 (Israeli lottery); Souza, Holos 2017 (Mega-Sena randomness).
