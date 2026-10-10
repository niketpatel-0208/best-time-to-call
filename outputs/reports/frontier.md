# Operating frontier - P6b(1)

Baseline: 124,782 calls, 7,383 meetings, 16.9 calls per meeting. All points use C1+C2 timing and the +/-20% per-band capacity constraint; the combined accounting (DR expected meetings over kept calls + credited deferred cool-off meetings).

| tau | cool-offs | expected meetings | net lift vs current | calls per meeting | meetings per 1,000 calls |
|---|---|---|---|---|---|
| 0 | off | 7697 | +4.3% | 16.2 | 61.7 |
| 0.005 | off | 7650 | +3.6% | 16.0 | 62.5 |
| 0.0075 | off | 7654 | +3.7% | 16.0 | 62.5 |
| 0.01 | off | 7495 | +1.5% | 15.2 | 65.8 |
| 0.0125 | off | 7348 | -0.5% | 14.4 | 69.7 |
| 0.015 | off | 7348 | -0.5% | 14.4 | 69.7 |
| 0.02 | off | 7029 | -4.8% | 12.9 | 77.3 |
| 0.025 | off | 6866 | -7.0% | 12.4 | 80.6 |
| 0.03 | off | 6571 | -11.0% | 11.4 | 87.4 |
| 0.04 | off | 6200 | -16.0% | 10.6 | 94.6 |
| 0.05 | off | 6040 | -18.2% | 10.2 | 98.1 |
| 0 | on | 7913 | +7.2% | 15.8 | 63.4 |
| 0.005 | on | 7868 | +6.6% | 15.6 | 64.2 |
| 0.0075 | on | 7867 | +6.6% | 15.6 | 64.2 |
| 0.01 | on | 7692 | +4.2% | 14.8 | 67.4 |
| 0.0125 | on | 7514 | +1.8% | 14.1 | 71.1 |
| 0.015 | on | 7514 | +1.8% | 14.1 | 71.1 |
| 0.02 | on | 7158 | -3.1% | 12.7 | 78.5 |
| 0.025 | on | 6978 | -5.5% | 12.2 | 81.7 |
| 0.03 | on | 6632 | -10.2% | 11.3 | 88.1 |
| 0.04 | on | 6242 | -15.5% | 10.5 | 95.2 |
| 0.05 | on | 6056 | -18.0% | 10.2 | 98.4 |

## Named operating points (95% seller-bootstrap CI)

| point | tau | cool-offs | expected meetings | net lift | 95% CI | calls per meeting | 95% CI |
|---|---|---|---|---|---|---|---|
| Max meetings | 0 | on | 7913 | +7.2% | [+5.9%, +8.4%] | 15.8 | [15.5, 16.1] |
| Max efficiency (= row f) | 0.0125 | on | 7514 | +1.8% | [+0.5%, +3.0%] | 14.1 | [13.8, 14.3] |

- **Max meetings** (tau=0, cool-offs on): the best net lift, +7.2%, at 15.8 calls per meeting.
- **Max efficiency** (tau=0.0125, cool-offs on, = row f): 14.1 calls per meeting at +1.8% net lift.
- The frontier is monotone: stronger stopping lowers calls per meeting and, past a point, lowers net lift; cool-offs shift the curve left (better efficiency) at a small lift cost.
