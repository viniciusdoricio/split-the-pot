# Roadmap

Ordered by how much each item could change the money you'd get *if* you win. None of them changes the odds of winning.

## 1. Measure the random-bet floor instead of guessing it (done, partly)
Result (README, "Random-bet floor"): f = 0 is ruled out and the floor is large (0.4-0.7), but quadra and quina disagree (0.51 vs 0.68), so the share of random bets isn't identified. Next: find what flattens quina crowding beyond random bets (multi-number bets, bolões) before changing the tool's default.

Original plan:
The web tool assumes 25% of bets are pure random (Surpresinha, random bolões). That number decides how empty the emptiest ticket can really be, and it moves the best-case return between about 65% and 89%.
- Fit a mixture model: E[W4] = B · P4 · (f + (1−f) · exp(Σβ)) for f and the betas jointly (nonlinear Poisson / quasi-likelihood; scipy.optimize or statsmodels with a custom link).
- f is identified by the draws made of the least popular numbers. Check the profile likelihood for f, then refit the jackpot validation with it.

## 2. Combination-level popularity, not just number-level
Today a ticket's crowd = product of number weights. Real crowds also come from shapes.
- Add features: diagonals, crosses and borders on the 6×10 slip, past winning combos replayed, arithmetic progressions, "all from one column/row", birthdays as day+month pairs (e.g. 12 with 25).
- Quina winners (5-subsets) carry more combination information than quadras: model W5 with pairwise interactions under L1/L2 shrinkage (1,770 pairs, 2,000 draws, so regularise and cross-validate).
- Validate on the jackpot counts (out/c_results.json → sena_validation): the new model must predict sena winners better than the current one.

## 3. Virada-specific crowd
Virada bettors are a different population (casual players, office bolões). The 2025 Virada had 6 winners where the model said ~16.
- Hierarchical model: base popularity + Virada deviation, shrunk toward zero (17 Viradas + the Mega 30 anos special, contest 3010).
- Question to answer: is the 2025 miss luck (Poisson p ≈ 0.01) or a systematic Virada shift?

## 4. Uncertainty you can trust (Bayesian)
- Refit the popularity model in PyMC (Poisson or negative binomial, offset log sales, period effects).
- Output for any ticket: a credible interval for expected co-winners, not just a point estimate.

## 5. Designing a set of tickets, not single tickets
For a set of k tickets, choose them jointly:
- each ticket low-crowd;
- pairwise overlap ≤ 2, so two of them can never split a quina;
- compare against Caixa multi-number bets (a 7-number bet = 7 combos sharing 5 numbers each, which means self-splitting quinas) and against bolões (35% fees).
- Formulate as integer programming or greedy + local search; report the EV per R$.

## 6. A "is the machine still fair?" monitor
- Sequential tests (CUSUM on per-number counts, rolling χ² with Monte Carlo p-values) over the last 100-500 draws.
- A dashboard in the BI tool of choice from data/draws.csv. Fine as a portfolio dashboard, even if it should keep saying "fair".

## Not worth time (already tested, no edge)
Hot/cold/overdue numbers, repeat-last-draw, sum or even/odd "ideal ranges", LSTM/GBM next-draw predictors, "quantum" anything. If one of these is revived, it goes through the walk-forward harness with multiple-testing correction first.
