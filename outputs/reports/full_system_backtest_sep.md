# Full-system backtest on September 2026: timing model + LLM callback layer

**Data**
- All 124,782 September calls, including 67,008 answered calls with a summary.
- The baseline is the current schedule: **7,383 meetings, 16.9 calls per meeting**.

**What was run**
- Every summary went through the `callback-time-extraction` rules, which look for:
  - an explicit time;
  - a hint;
  - wrong number;
  - do-not-call.
- Each next call was then decided by the `btc-callback-scheduler` priority order: do-not-call / wrong number → seller's time → seller's hint → model window → retry rules.
- The model-only numbers come from `backtest_report.md`.

## 1. What the reader found (September)
| | Calls |
|---|---|
| Answered calls with a summary | 67,008 |
| Call Later calls | 10,654 |
| ... wrong numbers (no callback) | 1,374 (13%) |
| ... seller gave an explicit time | 3,954 |
| ... seller gave a hint | 1,244 |
| Do-not-call requests | 483 |
| Follow-up calls (not Meeting Fixed) with a seller time or hint | 6,875 |
| ... of which a next call exists in the data | 3,715 |

**Reader accuracy.** An independent reader labelled a random sample of 200 summaries by hand (120 Call Later, 40 General, 40 Not Interested):
- found a timing cue: agreement 91%, precision 94%, recall 81%;
- type of cue: 89% agreement;
- same day: 89% agreement (n = 53);
- wrong number: 97% agreement;
- do-not-call: 98.5% agreement.

Recall is 81%, so the reader misses some cues, and the gain below is conservative.

## 2. What timing does to the next call
Meeting rate on the next call, depending on when it happened relative to the time the seller gave:

| Seller signal | Called **before** the seller's time | Called **at or after** it |
|---|---|---|
| Explicit time | 4.53% (n = 1,169) | **6.78%** (n = 1,563) |
| Hint (day, part of day, "after Diwali", travelling, unwell) | 2.89% (n = 693) | **6.55%** (n = 290) |

**Finding:** what costs meetings is **calling before the seller said they would be free**. 1,862 next calls in September (50% of the cued follow-ups) went out early. Calling exactly on time does no better than calling a little later, so the rule that matters is never to call early. The scheduler already does this: it uses the seller's time and never anything earlier.

## 3. Combined result vs the current schedule (September)
Assumptions:
- the early calls are moved to the seller's time;
- model-only is credited with its own +17.4% lift on those calls (conservative);
- calls to wrong numbers and do-not-call sellers are removed: 631 calls saved, 7 meetings lost.

| Setting | System | Meetings | vs today | 95% CI | Calls per meeting | Meetings per 1,000 calls |
|---|---|---|---|---|---|---|
| Max meetings | Model only | 7,913 | +7.2% | +5.9% to +8.4% | 15.8 | 63.3 |
| Max meetings | **Model + LLM layer** | **7,945** | **+7.6%** | +7.2% to +8.0% * | **15.7** | **63.9** |
| Max efficiency | Model only | 7,514 | +1.8% | +0.5% to +3.0% | 14.1 | 70.9 |
| Max efficiency | **Model + LLM layer** | **7,546** | **+2.2%** | +1.8% to +2.6% * | **14.0** | **71.7** |

\* The CI covers the uncertainty of the LLM layer's increment only: +39 meetings, 95% CI +9 to +70, from 400 resamples of sellers. The model's own uncertainty is in its row.

## 4. Reading
- **The model remains the main driver** (+7.2%). The LLM layer adds **+0.4 points** of meetings and saves **631 calls** a month, and it never makes things worse.
- **The LLM layer's gain is small in total because it only affects** follow-ups where the seller spoke about timing: 3,715 calls, 3% of September volume.
- **On the calls it does affect, the effect is large.** A seller called before their stated time converts at 2.9–4.5%; called at or after, at 6.6–6.8%.
- **Wrong numbers:** 13% of September "Call Later" calls were wrong numbers. They are now never called back.

## 5. Using the rest of the summaries: what the next call is worth
We classified each of the 59,592 answered, non-meeting September summaries. For every class, the table shows the actual outcome of the later calls in the same episode:

| What the summary says | Calls | Later calls | Meetings on them | Meeting rate | Calls per meeting |
|---|---|---|---|---|---|
| Seller gave a time / hint | 6,641 | 4,940 | 195 | 3.9% | 25 |
| Busy, no time | 2,668 | 2,444 | 76 | 3.1% | 32 |
| Already served (executive visited, already a member) | 1,146 | 741 | 17 | 2.3% | 44 |
| Positive interest, no time | 1,152 | 826 | 18 | 2.2% | 46 |
| No clear signal | 42,926 | 32,213 | 612 | 1.9% | 53 |
| **Hard no** | 3,123 | 1,713 | 22 | **1.3%** | **78** |
| **Wrong number** | 1,477 | 275 | 3 | **1.1%** | **92** |
| **Do not call** | 459 | 50 | 1 | 2.0% | 50 |

All later calls after an answered, non-meeting call: 43,202 calls (35% of September volume), 944 meetings, **45.8 calls per meeting**, compared with 16.9 overall.

**The summary adds information the call label doesn't have.** Within the same label:
- "Not Interested" calls whose summary says *busy*: next calls convert at **5.2%**.
- "Not Interested" calls whose summary says *hard no*: **1.2%**.
- Call Later calls with a *time or hint*: **4.5%**.
- Call Later calls with a *hard no*: **0%** (93 calls).

**Suppression rule, measured on actual outcomes.** Stop the remaining calls in an episode once the summary says wrong number, do-not-call or hard no:
- **2,038 calls saved (1.6%), 26 meetings lost (0.35%).**
- The removed calls convert at 1.3%, against 5.9% on average.
- This is limited to the same episode on purpose: a wrong-number seller's *new* hot leads still convert at 2.3%, so they keep being called.

## 6. Final combined result (September)
| Setting | Model only | Model + LLM layer (timing + suppression) |
|---|---|---|
| Max meetings | 7,913 meetings (+7.2%), 15.8 calls per meeting | **7,926 (+7.4%, CI +6.9% to +7.8%), 15.5 calls per meeting, 1.6% fewer calls** |
| Max efficiency | 7,514 (+1.8%), 14.1 calls per meeting, 70.9 per 1,000 calls | **7,527 (+2.0%), 13.8 calls per meeting, 72.4 per 1,000 calls, 1.9% fewer calls** |

**The ceiling, stated honestly:**
- The LLM layer acts only on calls where a person spoke: 55% of calls. Unanswered calls have no transcript.
- Its meeting gain comes from not calling early, about +40 to +52 meetings a month. That's large on the 1,862 calls it changes (from 3.9% to about 6.7%), but small against 7,383 meetings a month.
- Its second benefit is efficiency and seller experience: about 2,000 fewer unwanted calls a month.

## Caveats
- This is a historical-rate comparison, not a randomised test. The before vs at/after groups may differ in other ways.
- 3,160 cued follow-ups have no next call inside September, so their effect isn't counted.
- In production the voice agent's post-call fields (Sarvam LLM) do the extraction. Here the same rules were applied to the English summaries in batch.

Files:
- `sep_callback_extractions.csv` (internal, not included): one row per answered September call, with the extracted fields.
- `full_system_backtest_sep.md`: this report.
