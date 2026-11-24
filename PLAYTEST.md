# December plan (10-31 Dec)

## What playing regular draws can tell you
Regular Mega-Sena draws between 10 and ~27 December: roughly 6-9 draws. Caixa sometimes pauses or merges regular draws in the run-up to the Virada, so check the official calendar.

If you play **10 tickets per draw for 8 draws (80 tickets, R$480)**:
- expected quadras: 80 / 2,332 ≈ **0.03**, so about a 97% chance of zero quadras;
- expected quinas: 80 / 154,518 ≈ 0.0005;
- expected senas: 80 / 50,063,860 ≈ 0.0000016.

Zero prizes is the expected result for any ticket, good or bad, and a quadra would just be luck. So your own tickets can't test the strategy, and neither outcome should change the Virada plan.

## What can be tested for free: the model's predictions
The popularity model makes a falsifiable prediction for every draw: how many quadra and quina winners it should have, given its sales and its six numbers. That's an out-of-sample test that costs nothing.

**Pre-registered protocol (freeze on 10 Dec, before seeing those draws):**
1. On 10 Dec, run `run_all.sh`, then commit and tag `freeze-2026-12-10`. No refitting after this.
2. For each new draw d from 10 Dec on: predicted W4 = exp(fitted period effect for the latest quarter) · sales_d · exp(Σβ over the drawn numbers). The same for W5.
3. Baseline: the same formula with all β = 0 (no popularity).
4. Score: mean absolute log error of model vs baseline over all new draws, plus whether model error < baseline error draw by draw (sign test).
5. **Success** = the model beats the baseline in most draws and on mean log error. Write the result in this file before 31 Dec, good or bad.

## If you also play (optional, entertainment)
- **Budget, fixed now: R$ ____ total for December, Virada included.** Not topped up.
- Use low-crowd tickets from the tool, with fresh ones per draw.
- Log every ticket in `playtest_log.csv`. Compare total hits to the expected numbers above and nothing more.

## Virada decision (by 30 Dec)
- Tickets: generated in the tool (bottom 1-5%, overlap ≤ 2, not published anywhere).
- Number of tickets = what's left of the budget ÷ 6. Not more.
