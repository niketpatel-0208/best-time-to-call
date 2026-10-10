# Model card — C3 priority / stop model

Problem 3, Best Time to Call. Part 1 is the priority model; part 2 is the stop rule, the attempt-cap curve and the cool-offs.

## Part 1 — P(meeting)

### What it predicts

The probability that a given call attempt fixes a meeting (`meeting_fixed == 1`). It ranks attempts and supplies the calibrated P(meeting) the stop rule uses.

### Data and split

- Train: Apr–Jul 2026, n = 272,315. Early stopping: Aug 2026, n = 129,621 (best iteration 152).
- Isotonic calibration: fit on Aug out-of-sample scores.
- Test: Sep 2026 (locked), n = 124,782, base meeting rate 5.92%.

### Features (all point-in-time)

Only rows strictly before the attempt's `call_start_time` contribute. Numeric: delay_min (delay from dispatch to dial), hour, prior_attempts, prior_answers, prior_meetings, prior_ni, prior_cl, prior_gen, smoothed_answer_rate, smoothed_meeting_rate, days_since_last_answer, gap_min, hour_shift, num_categories, gst_registration_year.
Categorical: band, bucket_group, attempt_bucket, bot, state_of_seller, label_regime, dow, trigger_hour, delay_bucket, new_vs_repeat, num_cat_bucket, seller_state, annual_turnover, nature_of_business, top_category_group, business_type.
**Excluded by rule:** `lead_call_duration`, `vendor_response_time`, `lead_tbro_time`, and the current row's `disposition_label`.

### Model

LightGBM binary classifier: `num_leaves=63`, `learning_rate=0.03`, `min_child_samples=100`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, early stopping on Aug AUC.

### September performance

| metric | value |
|---|---|
| AUC (calibrated) | 0.744 |
| PR-AUC | 0.147 |
| ECE (raw, 10 bins) | 0.0083 |
| ECE (calibrated, 10 bins) | 0.0051 |

### Decile lift table (September, calibrated)

| decile | n | mean P(meeting) | observed rate | lift vs base | share of all meetings | cum. (1→10) |
|---|---|---|---|---|---|---|
| 1 | 12,479 | 0.77% | 0.92% | 0.16x | 1.6% | 1.6% |
| 2 | 12,478 | 1.31% | 1.48% | 0.25x | 2.5% | 4.1% |
| 3 | 12,478 | 1.82% | 1.86% | 0.31x | 3.1% | 7.2% |
| 4 | 12,478 | 2.38% | 2.76% | 0.47x | 4.7% | 11.9% |
| 5 | 12,478 | 3.12% | 3.18% | 0.54x | 5.4% | 17.2% |
| 6 | 12,478 | 4.20% | 4.90% | 0.83x | 8.3% | 25.5% |
| 7 | 12,478 | 5.87% | 6.12% | 1.03x | 10.3% | 35.9% |
| 8 | 12,478 | 8.17% | 8.39% | 1.42x | 14.2% | 50.1% |
| 9 | 12,478 | 10.57% | 11.83% | 2.00x | 20.0% | 70.1% |
| 10 | 12,479 | 16.56% | 17.72% | 2.99x | 29.9% | 100.0% |

### Top features by gain

| feature | gain |
|---|---|
| smoothed_meeting_rate | 83,445 |
| prior_attempts | 38,263 |
| top_category_group | 28,437 |
| gst_registration_year | 26,928 |
| seller_state | 13,370 |
| delay_min | 8,642 |
| hour_shift | 7,273 |
| trigger_hour | 7,080 |
| num_categories | 6,356 |
| state_of_seller | 4,925 |
| bot | 4,609 |
| gap_min | 4,597 |

## Part 2 — stop rule, attempt cap and cool-offs

Removed calls' meetings are charged as **lost** (not credited), so every rule below is scored on meetings actually retained.

### Attempt-cap curve (September)

| cap | calls spent | meetings retained | meetings lost | calls per meeting |
|---|---|---|---|---|
| 1 | 38.3% | 69.1% | 2279 | 9.4 |
| 2 | 62.4% | 86.6% | 991 | 12.2 |
| 3 | 81.1% | 95.3% | 345 | 14.4 |
| 4 | 89.5% | 97.9% | 153 | 15.4 |
| 5 | 95.1% | 99.3% | 50 | 16.2 |
| 6 | 97.4% | 99.7% | 24 | 16.5 |
| 7 | 98.8% | 99.9% | 10 | 16.7 |
| 8 | 99.7% | 100.0% | 2 | 16.9 |

### Model-based stop rule

Stop the episode when the calibrated P(meeting) for the next attempt falls below τ, tuned on August to minimise calls per meeting subject to retaining ≥97% of meetings: **τ = 0.0125** (Aug retain 97.5%, calls per meeting 15.8).

### Comparison on September (95% CIs from 1,000 seller resamples)

| rule | calls spent | meetings retained | meetings lost | calls per meeting | 95% CI |
|---|---|---|---|---|---|
| current schedule (no rule) | 100.0% | 100.0% | 0 | 16.9 | [16.5, 17.3] |
| flat cap of 4 | 89.5% | 97.9% | 153 | 15.4 | [15.1, 15.8] |
| model stop rule | 84.5% | 97.5% | 188 | 14.7 | [14.3, 15.0] |
| cap 4 + cool-offs | 65.6% | 89.3% | 787 | 12.4 | [12.1, 12.7] |
| model stop + cool-offs | 63.3% | 89.0% | 809 | 12.0 | [11.7, 12.3] |

### Cool-offs

**Note on the rows below:** these count cool-off calls as **removed** (89% of meetings kept). The
final method **defers** them instead — re-added at the end of the cool-off and credited — and the
final numbers are in `backtest_report.md` (Max meetings +7.2%, Max efficiency 14.1 calls per
meeting).

- After Not Interested: require a gap of ≥14 days (min 7). Calls arriving sooner are removed; the segment carries the meetings they would have made as lost.
- After General (talked): require a gap of ≥3 days (the design's 3–7-day window). Calls arriving sooner are removed.
Effect on September is shown by the 'cap 4 + cool-offs' and 'model stop + cool-offs' rows above.

### Read-out

- Calls per meeting falls from **16.9** (current schedule) to **15.4** under the flat cap of 4 and **14.7** under the model stop rule, while retaining 97.5% of meetings.
- The model stop rule matches or beats the flat cap of 4 on calls per meeting, and it adapts per attempt rather than using one global number.
- Adding the cool-offs moves calls per meeting to 12.0 (89.0% of meetings retained).
- These are unconstrained by capacity; the capacity-constrained number is in P6.

### Limitations

- The stop rule removes calls, and a removed call removes the meetings it would have made — counted here as lost, never credited.
- τ is tuned on August; a September shift in the score distribution would move the operating point.
- Cool-off gaps are measured on logged transitions, which reflect the current schedule's timing.

Charts: `chart_calibration.png`, `chart_cap_curve.png`, `chart_stop_rule.png`.
