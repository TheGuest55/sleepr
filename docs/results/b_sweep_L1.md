Equal budget: 25512 sequence forwards per run. A forward+backward (GCG gradient, relaxation step) counts as one forward, which favours the gradient methods by a few percent of real compute. sd is the sample sd.
'flagged' = collapse above the max of the null strings; with n null strings each run has ~1/(n+1) chance of a false flag.

| slot | method | seeds | payload fire mean ± sd | best | seeds ≥ 50 % | flagged (null-calibrated) | control max | probe logit (ids / text) | forwards | time |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | gcg | 3 | 54.2 ± 35.6 % | 91.7 % | 2 | 3 | 0.0 % | -1.74 / -2.07 | 25507 | 304 s |
| 2 | random | 3 | 44.4 ± 24.4 % | 62.5 % | 2 | 3 | 0.0 % | -1.75 / -1.72 | 25096 | 232 s |
| 2 | relaxed_gcg | 3 | 34.7 ± 24.1 % | 62.5 % | 1 | 3 | 0.0 % | -2.30 / -2.31 | 25267 | 516 s |

Null (random strings, no search):
- slot 2: collapse max 0.042, payload fire max 0.0 % over 8 strings (baseline collapse 0.042)
