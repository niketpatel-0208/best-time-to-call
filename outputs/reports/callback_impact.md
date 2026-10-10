# Callback scheduling — impact (B2)

Schedules the next call for all 2,000 Call-Later calls with the
`btc-callback-scheduler` skill and measures the gap against what the system
actually did. Inputs: `callback_extractions.csv`, `seller_windows.csv` (per-seller
model bands) and `retry_policy.json`; the system's behaviour comes from
`attempts.csv`.

Every row is disposition *Call Later / Busy*. `callbacks_so_far` is taken as 0
(first callback in the episode) and the outcome is Call Later for all rows, so
rules 1–5 fire (wrong_number / do_not_call / seller_stated / seller_hint /
model_window).

## 1. Source mix

**Wrong numbers are removed first.** 381 rows are wrong-number / "not from that
company" calls: the agent reached someone who was not the seller, so there is no
seller and no callback. They now carry source **`wrong_number`** (no next call)
and are excluded from the callback population.

| source | n | share of 2,000 | meaning |
|---|---|---|---|
| `wrong_number` | 381 | 19.1% | not the seller — no callback, excluded |
| `seller_stated` | 632 | 31.6% | an explicit time the seller gave |
| `seller_hint` | 385 | 19.3% | an implicit hint (daypart, "out till…", next week) |
| `model_window` | 600 | 30.0% | no usable cue — the model's band decides |
| `do_not_call` | 2 | 0.1% | seller asked never to be called again |

On the **callback population** (2,000 − 381 = **1,619** rows): seller_stated
632 (39.0%), seller_hint 385 (23.8%), model_window 600 (37.1%), do_not_call 2
(0.1%). So **1,017 rows (62.8% of the callback population) get the seller's own
time or hint** instead of a default; the other 600 fall back to the model.

## 2. Safety limit in the scheduler

A seller-derived cue is rejected and the row falls back to `model_window` if the
computed `next_call_at` would be **more than 90 days after `call_time`**, or if
the cue **refers to a date before the call**.

- **29 rows affected** (23 were explicit, 6 implicit): **18** were >90 days out
  and **11** were past-date references.
- Example: a past-reference "previous conversation on April 26th" had produced a
  2027 date; it now falls back to the model's next band. The same rule drops a
  far-future festival cue (e.g. "Diwali") that is more than 90 days away.
- Effect on the mix: seller_stated 655→632 and seller_hint 391→385; model_window
  951→600 (the +29 fallbacks plus the 381 wrong numbers removed).

## 3. Seller-stated rows: our time vs what the system did

For the 632 `seller_stated` rows in the callback population (our target is the
stated time − 5 min):

- Rows with an actual next attempt: **557 of 632**.
- **Median gap: 0.9 h.  p90 gap: 159.2 h (~6.6 days).**
- **Called late by more than 1 hour: 42.5%.**

Restricting to the **precise** cues (clock / relative minutes-hours, n = 316):

- **Median 0.8 h · p90 110.1 h · late by >1 h: 47.2% · within ±30 min: 37.7%.**

Even where the seller gave an exact time, the system misses it by more than an
hour about half the time — consistent with the finding that the system ignores
most cues ("call in 10 minutes" got a callback after a median of ~19–20 h).

## 4. Expected effect (from historical rates — not a measured result)

> This is an **expected** estimate built by mapping each row's callback gap onto
> the historical meeting rates in `03_verified_findings.md` (≤1 h 7.0%,
> 1–6 h 6.6%, 6 h–1 d 4.4%, 1–2 d 4.5%). It is **not** a measured backtest.

Gap distribution and expected meetings per 1,000 call-later calls, on the
1,619-row callback population:

| callback gap | rate | system actual (n) | system meetings | our schedule (n) | our meetings |
|---|---|---|---|---|---|
| ≤1 h | 7.0% | 10 | 0.7 | 94 | 6.6 |
| 1–6 h | 6.6% | 96 | 6.3 | 530 | 35.0 |
| 6 h–1 d | 4.4% | 459 | 20.2 | 220 | 9.7 |
| 1–2 d | 4.5% | 342 | 15.4 | 250 | 11.3 |
| >2 d | 4.4% * | 522 | 23.0 | 523 | 23.0 |
| no next call | 0% | 190 | 0.0 | 2 | 0.0 |
| **total** | | **1,619** | **40.5** | **1,619** | **52.8** |

\* the findings give no rate beyond 2 days; 4.4% (the lowest given) is used as a
placeholder. It is generous to the system — with the true >2-day rate likely
lower, the gap would widen, not shrink.

**Expected meetings per 1,000 call-later calls: ~41 under the system's actual
timing vs ~53 under our schedule — a lift of ~12 per 1,000 (+30%).** The lift
comes almost entirely from moving same-day callbacks into the 1–6 h window: our
schedule puts 530 rows there (33%), the system only 96 (6%). A sensitivity check
with the >2-day bucket at 2.0% gives 32.8 (system) vs 45.1 (ours) — the
direction and rough size of the lift are unchanged.

Two levers are not in the number above: (a) we call at the stated time − 5 min,
i.e. ~5 min early, which the findings put at 5.9% versus 3.8–4.0% for 5–60 min
late; and (b) the 190 calls the system never followed up at all become callbacks
under our schedule.

## 5. Why there is no September backtest

The September backtest is **skipped**: the Call-Later summaries — and therefore
the callback-time extraction this schedule is built on — exist only for
**April–May 2026**. There is no extracted cue for June–September, so the
callback policy cannot be replayed on the locked September split. The callback
effect therefore **stays an expected effect from historical rates**, not a
measured one. (The model's band/retry backtest is a separate deliverable and is
unaffected.)

## 6. Examples

| # | our reason (plain language) | what the system actually did |
|---|---|---|
| 1 | "Seller said 8–9 PM — call at 19:55 the same day." | Called **25 days later** (6 May, 16:44) → Not Interested. Stated window ignored. |
| 2 | "Seller said call in 10 minutes — call at 16:00." | Called **1 h 50 m later** (17:50) → General (talked). |
| 3 | "Seller is out of station — call the model's best band after they return (25 May, 19:00)." | Called **3 days early** (22 May, 11:59) → Not Answered. |
| 4 | "No usable cue — same day, model band 1–6 h later (11:00)." | Called **next day** (25 Apr, 11:51) → Not Interested. |
| 5 | "Wrong number — no callback (source `wrong_number`)." | Called **again the next day** (2 May, 11:53) → Not Interested. A wasted dial on a non-seller. |

## Caveats

- The system-side comparison uses the next attempt in `attempts.csv` for the same
  seller; 190 of the 1,619 callbacks have no later attempt and count as 0.
- Section 4 is an **expected effect** from historical rates, not a measured
  result; it inherits the extractor's errors and the >2-day rate assumption.
- Band capacity (±20% per hour band) is not enforced — the schedule is the
  unconstrained target.
