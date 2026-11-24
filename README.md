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
| Is that real? | Stable 1998-2019 vs 2020-2026 (r = 0.93); predicts jackpot winner counts (exponent 1.63 ± 0.14; theory is ~1.59 once the two unmatched numbers of a quadra are counted, not 1.5) | out/b_results.json, out/c_results.json, out/f_results.json |
| How many bets are effectively random? | A lot: the floor is around 0.4-0.7, not the 25% guess, and 0 is ruled out. The exact share isn't identified (quadra and quina disagree) | out/f_results.json |
| What's a ticket worth at the Virada? | Average R$2.14 per R$6. Least-crowded: about R$3.0-4.4 with ~5-7 co-winners (B = 520M). The old R$5.35 assumed no random bets, which the data rules out | out/f_results.json, out/c_results.json |

## Layout
```
src/        pipeline scripts (see Scripts below)
out/        generated results (json/csv/pkl), recreated by run_all.sh
data/       draws.csv, flat file for Excel / Power BI
web/        tool_template.html + the built split_the_pot.html
ROADMAP.md  next analyses worth doing
PLAYTEST.md the 10-31 Dec plan and pre-registered model test
```

## Random-bet floor (ROADMAP 1)
Some bets are pure random (Surpresinha, random bolões), and they put a floor under how empty any ticket can be. The web tool guessed 25%. This tries to measure it.

**Model.** A share f of bets is uniform; the rest pick combination c with probability ∝ Π w_i over its six numbers. That gives exact expected quadra, quina and sena counts for any draw (elementary symmetric polynomials, `src/floor_model.py`), with one set of 60 weights for all tiers. Quarter effects and a Virada offset as in `b_popularity.py`. Fitted on quadra + quina counts (quasi-Poisson); sena counts are only used for validation.

**Can f be recovered at all?** First a simulation on the real draws and sales with noise matched to the data (`src/f_floor_sim.py`, 50 reps per case). Yes, if the model is right: at true f = 0.25 the estimate is unbiased (0.253), 95% CIs are 0.18 wide and cover 98% of the time. Adding slip-pattern effects the model doesn't have (true f = 0) gives a false f > 0.10 in 12% of reps.

**Real data** (2,020 draws):
| Fit | f | 95% CI |
|---|---|---|
| quadra + quina (primary) | 0.51 | 0.46-0.55 profile; 0.38-0.58 bootstrap by year |
| quadra only | 0.51 | 0.43-0.57 (Bonferroni, 4 tests) |
| quina only | 0.68 | 0.58-0.76 (Bonferroni) |
| before 2015 / 2015 on | 0.46 / 0.54 | 0.31-0.56 / 0.46-0.60 |

f = 0 is rejected hard (deviance 130). But the pre-registered rule 4 failed: quadra and quina give non-overlapping answers. So f is not identified as "the share of random bets". Something else also flattens the crowd, more at the quina level (multi-number bets and bolões are candidates, untested). The number is a model parameter, not a measured share.

**Jackpot check** (10 time blocks, popularity fitted without the held-out block, sena log score vs the current model): +19.5 (year-bootstrap CI 1.4 to 46.4), so it passes. Most of that is the Viradas (+14.7) and one draw alone (contest 2110, 52 winners) is +10.1. Without the Viradas: +4.8 (CI -2.1 to 12.0). Read it as "at least as good", not as proof of f; the simulation shows sena counts can barely tell f values apart.

**In reais** (least-crowded 1% ticket, B = 520M combos; average ticket stays at R$2.14 and ~10.4 co-winners):
| f | co-winners | EV per R$6 |
|---|---|---|
| 0 (ruled out) | 3.7 | R$5.72 |
| 0.38 (bootstrap low) | 5.0 | R$4.37 |
| 0.51 (estimate) | 5.7 | R$3.85 |
| 0.58 (bootstrap high) | 6.2 | R$3.55 |
| 0.70 (quina-only) | 7.3 | R$3.03 |

With 620M combos each EV is R$0.5-0.7 lower. Note the old "conservative" row (f = 0.25 on top of betas fitted assuming f = 0) counted the floor twice; refitting the weights at each f is what this table does. `c_jackpot.py` and the web tool still use the old 25% until this is reviewed.

## Scripts
- `src/fetch_data.py`: downloads all draws (winners per tier and sales) and cross-checks the numbers against a second scrape. Writes `out/draws.pkl` and `data/draws.csv`.
- `src/a_randomness.py`: Monte Carlo fairness tests.
- `src/d_predict.py`, `src/d2_hot.py`: walk-forward strategy backtests and ML.
- `src/b_popularity.py`: number popularity GLM and pattern effects. Writes `out/popularity.csv`.
- `src/c_jackpot.py`: jackpot validation and Virada EV scenarios.
- `src/e_generator.py`: low-crowd tickets and the model export for the web tool.
- `src/build_tool.py`: writes `web/split_the_pot.html` (self-contained, open in a browser).
- `src/floor_model.py`, `src/floor_data.py`: structural popularity model with a random-bet floor.
- `src/f_floor.py`: estimates the floor, jackpot validation, EV by f. Writes `out/f_results.json` (~10 min).
- `src/f_floor_sim.py`: can f be recovered? Writes `out/f_sim.json` (~20 min; `bash run_all.sh --sim`).
- `tests/`: exactness and gradient checks for the floor model (`python -m pytest tests`).

Sales exist from 2009 on (plus a few 1998 draws). Bet prices before 2015 are approximate and only used to turn sales into bet counts.

## Sources
- Draws, winners and sales (scraped from Caixa): github.com/GabrielRossiSoares/megasena; cross-checked with github.com/sosthenis/megasena-data, github.com/guilhermeasn/loteria.json, github.com/mneves75/megasena-analyser-webapp (0 mismatches).
- Method references: Farrell & Walker on the UK lottery (inferring popularity from winner counts); Number preferences in lotteries, JDM 2016 (Dutch Lotto); JDM 2021 (Israeli lottery); Souza, Holos 2017 (Mega-Sena randomness).
