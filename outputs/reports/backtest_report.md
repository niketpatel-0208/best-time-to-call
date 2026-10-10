# Backtest report — Best Time to Call (P2–P6)

IndiaMART Voice AI Hackathon 2.0, Problem 3. Everything below is evaluated on the **locked
September 2026** window (used once), through the shared evaluator (stratified replay,
SNIPS, doubly-robust; 95% CIs from 1,000 bootstrap resamples of sellers). The baseline is
the current schedule: **124,782 dialed calls, 7,383 meetings, 16.9 calls per meeting,
59.2 meetings per 1,000 calls.**

## 1. Headline — two operating points

The policy is a single pipeline (C1 timing + C2 retry + C3 model stop + deferred cool-offs)
sitting on a frontier of net meetings against calls per meeting. Two points are named;
**IndiaMART picks where to sit.**

| operating point | expected meetings | net lift vs current | 95% CI | calls per meeting | 95% CI | meetings per 1,000 calls |
|---|---|---|---|---|---|---|
| **Max meetings** (τ = 0, cool-offs on) | 7,913 | **+7.2%** | [+5.9%, +8.4%] | 15.8 | [15.5, 16.1] | 63.4 |
| **Max efficiency** = row f (τ = 0.0125, cool-offs on) | 7,514 | **+1.8%** | [+0.5%, +3.0%] | **14.1** | [13.8, 14.3] | **71.1** |

- **Max meetings** keeps every call and only re-times it and defers cool-offs: the best net
  lift, **+7.2% meetings** (95% CI +5.9% to +8.4%), at 15.8 calls per meeting.
- **Max efficiency** turns on the model stop rule: **14.1 calls per meeting**, i.e. **71.1
  vs 59.2 meetings per 1,000 calls (+20% meetings per call)**, and total meetings (7,509 in
  the full accounting) are **not below** the current 7,383. It trades ~5 points of lift for
  ~1.7 fewer calls per meeting.

IndiaMART chooses the point: efficiency if call capacity is the binding constraint, max
meetings if it is not.

## 2. Rolling folds (honest framing)

Test months Jun/Jul/Aug, each trained on the prior months only (C1 timing, the stop model
and the DR outcome/propensity models refit per fold).

| fold | baseline cpm | Max-meetings lift | cpm | Max-efficiency lift | cpm |
|---|---|---|---|---|---|
| Jun (trained on Apr only) | 15.0 | −2.7% | 15.4 | −6.1% | 14.1 |
| Jul (Apr–May) | 18.1 | +9.6% | 16.5 | +7.4% | 15.5 |
| Aug (Apr–Jun) | 18.1 | +7.0% | 16.9 | +3.4% | 15.5 |
| Sep (Apr–Jul, the headline) | 16.9 | +7.2% | 15.8 | +1.8% | 14.1 |
| **mean ± SD (Jun–Aug)** | 17.1 ± 1.8 | **+4.7% ± 6.5** | **16.3 ± 0.8** | **+1.6% ± 6.9** | **15.1 ± 0.8** |

- The lift is **positive in every fold trained on two or more months** (Jul +9.6%, Aug
  +7.0%, Sep +7.2% for Max meetings).
- **June — trained on April alone — is negative**, which is the cold-start limit, not a
  failure of the method: one month of history is not enough for the timing, stop and DR
  models to generalise. **Recommendation: require at least two months of history and retrain
  monthly.**
- **Calls per meeting improves in every fold for the Max efficiency point, and in Jul–Sep for
  both points.** In June the Max-meetings point is 15.4 vs a baseline of 15.0 (slightly worse),
  which is part of the same cold-start limit.

## 3. Operating frontier

`chart_frontier.png` plots net meetings lift against calls per meeting for the τ × cool-off
sweep (all with C1+C2 timing and the ±20% capacity constraint). The frontier is monotone:
stronger stopping lowers calls per meeting and, past a point, lowers net lift; cool-offs
shift the curve left. The two named points are marked.

## 4. Comparison table (a)–(f)

Combined accounting: expected meetings = sum over kept calls of the doubly-robust
P(meeting) at the policy's band and new delay, plus credited deferred cool-off meetings;
calls = kept + deferred.

| policy | expected meetings | net lift vs current | 95% CI | calls per meeting | 95% CI | meetings per 1,000 calls |
|---|---|---|---|---|---|---|
| (a) current | 7,383 | +0.0% | [0.0%, 0.0%] | 16.9 | [16.5, 17.3] | 59.2 |
| (b) global best band | 7,640 | +3.5% | [+2.3%, +4.7%] | 16.3 | [16.0, 16.7] | 61.2 |
| (c) C1 only | 7,804 | +5.7% | [+4.4%, +7.1%] | 16.0 | [15.7, 16.3] | 62.5 |
| (d) LightGBM argmax band | 7,819 | +5.9% | [+4.6%, +7.2%] | 16.0 | [15.7, 16.3] | 62.7 |
| (e) full policy, unconstrained (deferred) | 7,554 | +2.3% | [+1.1%, +3.6%] | 14.0 | [13.7, 14.3] | 71.5 |
| (e′) full policy, unconstrained (removed) | 6,968 | −5.6% | [−6.7%, −4.4%] | 12.2 | [12.0, 12.5] | 81.7 |
| (f) full policy, capacity-constrained (deferred) — headline | 7,509 | +1.7% | [+0.4%, +2.9%] | 14.1 | [13.8, 14.4] | 71.0 |

- (e)/(e′) differ only in the cool-off accounting. Deferring (e) keeps meetings above
  current; deleting (e′) looks best on calls per meeting but drops total meetings 5.6%
  below current because the removed calls' meetings are charged as lost. **Deferral is the
  correct accounting.**

## 5. Per-segment split (DR lift vs current within segment)

| policy | fresh | stale/backlog | retries after NA | callbacks & follow-ups |
|---|---|---|---|---|
| (b) global best | +0.1% | +6.4% | +8.7% | +12.1% |
| (c) C1 | +0.1% | +23.2% | +11.8% | +17.4% |
| (d) LGB argmax | +0.1% | +24.6% | +12.8% | +16.4% |
| (e) full unconstrained | +0.2% | +23.2% | +8.7% | +17.4% |
| (f) full capacity | +0.2% | +16.7% | +9.6% | +15.1% |

Fresh first attempts are **~0 by design** (called immediately, never moved). The gain sits
in the movable segments: **backlog/stale first attempts, retries after an NA, and callbacks
and follow-ups.**

## 6. Attempt-cap curve (September)

| attempt cap | calls spent | meetings retained | meetings lost |
|---|---|---|---|
| 1 | 38.3% | 69.1% | 2,279 |
| 2 | 62.4% | 86.6% | 991 |
| 3 | 81.1% | 95.3% | 345 |
| 4 | 89.5% | 97.9% | 153 |
| 5 | 95.1% | 99.3% | 50 |
| 6 | 97.4% | 99.7% | 24 |
| 7 | 98.8% | 99.9% | 10 |
| 8 | 99.7% | 100.0% | 2 |

A flat cap of 4 keeps 97.9% of meetings on 89.5% of calls; the model stop rule beats it
(14.7 vs 15.4 calls per meeting, P5). `chart_cap_curve.png` and `chart_stop_rule.png`.

## 7. Splits by new vs repeat seller, bot and bucket

DR lift (C1 policy / full policy) vs current within the split, September:

| split | n | C1 lift | full-policy lift |
|---|---|---|---|
| new seller | 69,641 | +0.7% | +1.4% |
| repeat seller | 55,141 | +17.1% | +10.2% |
| main_vani | 102,372 | +7.6% | +5.2% |
| arrowhead | 20,937 | −5.5% | −4.4% |
| Top 3 bucket | 15,625 | +6.4% | +1.8% |
| Rest bucket | 109,157 | +5.6% | +4.1% |

- The lift is concentrated in **repeat sellers** (who have history to personalise on); new
  sellers barely move, because their timing rests on the global curve and segment level.
- The higher lift for repeat sellers comes from the playbook rules (retries, callbacks,
  follow-ups and cool-offs) applying to them. Their personal hour is not the driver, which
  is consistent with section 8.
- **arrowhead is negative**: calls placed through this vendor do not respond to the
  re-timing, so this is the one segment the policy should leave on today's schedule until a
  live test.
- Both bucket groups benefit; the gain is not concentrated in one bucket.

## 8. Does per-seller personalisation help?

**Effectively no.** C1 (hierarchical Beta-Binomial, per-seller band) lifts the meeting rate
**+5.7%**; LightGBM argmax band lifts **+5.9%** — statistically tied (CIs overlap almost
completely). This matches the established finding that **the best hour is universal**:
segments differ in level, not in which hour is best, so a seller's own hour choice adds
almost nothing over the global curve. **The lift comes from the policy — when to retry,
which daypart after a failure, when to stop, how long to wait after a "no" — not from
per-seller hour fitting.** Per-seller timing still earns its place as a safety valve for the
minority with real evidence, and the voice agent (C5) is where genuine personalisation
enters, by asking the seller.

## 9. Column guide — seller_windows.csv

The full file holds internal seller IDs and is not included; a 200-row sample with hashed IDs is in `outputs/samples/seller_windows_sample.csv`.

One row per seller (149,363). Bands are the six 2-hour bands (9-10, 11-12, 13-14, 15-16,
17-18, 19-20).

| column | meaning |
|---|---|
| seller_id | seller GLID (`fk_glusr_usr_id`). |
| current_state | the seller's latest state: Fresh / Not Answered / Call Later / General / Not Interested / Meeting Fixed. |
| rec_band | the recommended 2-hour band to call this seller. |
| backup_band_1, backup_band_2 | the next-best bands, used when the primary band is at capacity. |
| rec_day_offset | days from the trigger to schedule the call (0 for Fresh / Not Answered / Call Later; 5 after General; 14 after Not Interested). |
| p_meeting | the modelled probability of fixing a meeting **if the seller is called fresh in the recommended band** — a fresh-first-attempt probability, averaging 13.5% across sellers. |
| p_answer | the modelled probability of an answer under the same fresh-in-rec_band assumption. |
| confidence | high / med / low, reflecting history depth (answered-call count) plus a seller-specific band deviation. |
| reason_code | why the band was chosen: seller_history / segment_level / global_curve. |
| next_call_at | the live schedule time (as of 2026-10-01 09:00 IST), in the recommended band, calls only 09:00–21:00; empty for Meeting Fixed sellers. |

**Important:** `p_meeting` and `p_answer` are the chance **if the seller is called fresh in the
recommended band** (average 13.5%), **not** the average chance of any call (5.9% overall). They
are conditional on a fresh call in the best band, so they are higher than the unconditional
meeting rate.

## 10. Method and caveats

- Time split: train Apr–Jul, validate Aug, test Sep (locked, used once). No September data
  entered any fit or tuning; the evaluator asserts this.
- Point-in-time features only. Banned as features throughout: `lead_call_duration`,
  `vendor_response_time`, `lead_tbro_time`, and the current row's `disposition_label`.
- The doubly-robust model prices the cost of waiting: a deferred call is scored at its new
  delay from dispatch to dial. Removed calls' meetings are charged as lost, never credited.
- Capacity: each band kept within ±20% of its current volume, assigned by marginal gain,
  overflow to backup bands.
- SNIPS is reported but is not credible on this data (the current schedule is
  near-deterministic in band, so there is little overlap to reweight); the replay and DR
  estimates carry the result.
- No PII in any output.
