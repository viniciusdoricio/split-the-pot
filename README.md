# Split the Pot

**Inferring which lottery numbers people pick from 30 years of winner counts.** Poisson GLM, Monte Carlo fairness tests, walk-forward backtests, and a ticket tool. Spoiler: you can't beat the odds, but you can avoid sharing the prize.

Built on all 3,089 draws of Brazil's Mega-Sena (1996–2026), aimed at the Mega da Virada 2026.

**Bottom line:** no model can change the 1-in-50,063,860 odds. Measured popularity of numbers can roughly halve how many people you'd share a jackpot with. At the Virada that is the difference between an expected ~R$96 mi and ~R$180–250 mi if you win. Expected value per R$6 ticket stays negative.

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
| Is the draw fair today? | Yes. 2023–2026 frequency test p = 0.64; all-time unevenness (p ≈ 0.004) comes from old eras | out/a_results.json |
| Do hot/cold/overdue/ML predict? | No out-of-sample edge in recent eras (ML AUC 0.500) | out/d_results.json, out/d2_results.json |
| Which numbers are crowded? | 13, 5, 7, 3, 10 and most of 1–31 (+8% on average); 40, 50, 41, 30 and 52–60 are the emptiest | out/popularity.csv |
| Is that real? | Stable 1998–2019 vs 2020–2026 (r = 0.93); predicts jackpot winner counts (exponent 1.63 ± 0.14 vs theory 1.5) | out/b_results.json, out/c_results.json |
| What's a ticket worth at the Virada? | Average R$2.14 per R$6; least-crowded R$3.90–5.35 depending on assumptions | out/c_results.json |

## Layout
```
src/        pipeline scripts (see CLAUDE.md for what each does)
out/        generated results (json/csv/pkl), recreated by run_all.sh
data/       draws.csv, flat file for Excel / Power BI
web/        tool_template.html + the built split_the_pot.html
ROADMAP.md  next analyses worth doing
PLAYTEST.md the 10–31 Dec plan and pre-registered model test
```

## Sources
- Draws, winners and sales (scraped from Caixa): github.com/GabrielRossiSoares/megasena; cross-checked with github.com/sosthenis/megasena-data, github.com/guilhermeasn/loteria.json, github.com/mneves75/megasena-analyser-webapp (0 mismatches).
- Method references: Farrell & Walker on the UK lottery (inferring popularity from winner counts); Number preferences in lotteries, JDM 2016 (Dutch Lotto); JDM 2021 (Israeli lottery); Souza, Holos 2017 (Mega-Sena randomness).
