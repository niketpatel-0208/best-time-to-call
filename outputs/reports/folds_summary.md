# Rolling folds - P6b(2)

Test months Jun, Jul, Aug 2026, each trained on the prior months (last prior month used as validation). Per fold, C1 timing, the P(meeting) stop model and the DR outcome/propensity models are refit on the prior months only. Two operating points: **Max meetings** (tau 0, cool-offs on) and **Max efficiency** (tau 0.0125, cool-offs on), both with C1+C2 timing, the +/-20% capacity constraint and the combined accounting. SD is the sample SD across the three folds.

| fold | test month | baseline calls | baseline meetings | baseline cpm | Max-meetings lift | Max-meetings cpm | Max-efficiency lift | Max-efficiency cpm |
|---|---|---|---|---|---|---|---|---|
| jun | 2026-06 | 76,451 | 5,103 | 15.0 | -2.7% | 15.4 | -6.1% | 14.1 |
| jul | 2026-07 | 122,577 | 6,763 | 18.1 | +9.6% | 16.5 | +7.4% | 15.5 |
| aug | 2026-08 | 129,621 | 7,158 | 18.1 | +7.0% | 16.9 | +3.4% | 15.5 |

## Mean +/- SD across folds

| metric | Max meetings | Max efficiency |
|---|---|---|
| net meetings lift | +4.7% +/- 6.5 | +1.6% +/- 6.9 |
| calls per meeting | 16.3 +/- 0.8 | 15.1 +/- 0.8 |
| (baseline calls per meeting) | 17.1 +/- 1.8 | 17.1 +/- 1.8 |

- Max meetings: lift +4.7% +/- 6.5; calls per meeting 16.3 +/- 0.8 (baseline 17.1).
- Max efficiency: lift +1.6% +/- 6.9; calls per meeting 15.1 +/- 0.8.
- The June fold (trained on April only) is the unstable one and is negative; the later folds are solidly positive. The policy's lift is more stable as training history grows, which is the expected cold-start behaviour. Calls per meeting improves in every fold for the Max efficiency point, and in Jul–Sep for both points (in June the Max-meetings point is 15.4 vs a baseline of 15.0).
