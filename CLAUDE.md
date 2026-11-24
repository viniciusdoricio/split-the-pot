# Split the Pot: instructions for Claude

## What this project is
A for-fun but rigorous data project on Brazil's Mega-Sena, aimed at the **Mega da Virada 2026** (contest 3100; bets close 22:00 on 31 Dec 2026, draw 11:00 on 1 Jan 2027, simple bet R$6, jackpot estimate ≥ R$1 bi, no rollover).

Owner: Vinicius, a data/BI analyst. Talk to him like a colleague: direct, numbers first, no hype.

## The ground truth already established (don't re-litigate without new evidence)
- Every 6-number combination has a 1 in 50,063,860 chance. Nothing in this repo changes that.
- Randomness: the 2023–2026 draws are indistinguishable from a fair lottery (out/a_results.json). The all-time frequency unevenness (p ≈ 0.004) comes from old eras.
- Prediction: hot, cold, overdue, repeat and gradient boosting all show no out-of-sample edge in recent eras (out/d_results.json, out/d2_results.json).
- The real lever is **popularity**. Inferred from quadra and quina winner counts with a quasi-Poisson GLM (out/b_results.json), it is stable over time (r = 0.93) and predicts jackpot winner counts it never saw (exponent 1.63 ± 0.14 vs 1.5 theory). Unpopular tickets expect about 4–6 co-winners at the Virada instead of about 10.
- Expected value stays negative for every ticket. Best case is about R$3.90–5.30 back per R$6.

## Rules for any new analysis
1. **Out-of-sample or it didn't happen.** Any "pattern" must beat random in a walk-forward test on draws it never saw. In-sample fits prove nothing.
2. **Count your tries.** When testing many windows, features or strategies, correct for multiple testing (Bonferroni or a permutation test over the whole search). State how many things were tried.
3. **Translate to reais.** Every result ends with what it does to the expected R$ per ticket and to the number of expected co-winners.
4. **Be a counterweight.** If Vinicius (or you) starts reading signal into noise, say so plainly, with the number. A small sample of draws or of his own tickets proves nothing. Say that before he spends money on it.
5. Never present any output as a way to "win" or as increasing the odds of hitting the numbers. Ticket quality = fewer people to share with, nothing more.
6. Keep spending decisions his. Don't encourage bigger budgets; the budget lives in PLAYTEST.md.

## How to run
```
pip install -r requirements.txt
bash run_all.sh            # fetch -> tests -> models -> tickets -> web tool
```
- `src/fetch_data.py`: downloads all draws (winners per tier and sales) and cross-checks against a second scrape. Writes `out/draws.pkl` and `data/draws.csv`.
- `src/a_randomness.py`: Monte Carlo fairness tests.
- `src/d_predict.py`, `src/d2_hot.py`: walk-forward strategy backtests and ML.
- `src/b_popularity.py`: number popularity GLM and pattern effects. Writes `out/popularity.csv`.
- `src/c_jackpot.py`: jackpot validation and Virada EV scenarios.
- `src/e_generator.py`: low-crowd tickets and the model export for the web tool.
- `src/build_tool.py`: writes `web/split_the_pot.html` (self-contained, open in a browser).

Data notes: sales exist from 2009 on (plus a few 1998 draws). Bet prices before 2015 are approximate and used only to turn sales into bet counts. Sources are listed in README.md.

## What to work on
See ROADMAP.md, in priority order. The plan for 10–31 December is PLAYTEST.md.
