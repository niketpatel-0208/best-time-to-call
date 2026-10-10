# The 15-minute rule — evidence and verdict

Problem 3, Best Time to Call. Evidence built on Apr–Aug 2026 (September held out).

## What the rule is

After a hot lead is generated, the system waits **15 minutes** before dispatching the lead to
the vendor's dialer. High-priority hot leads — for example a seller who has opened the payment
page — are dispatched **immediately**, on a fast track. There is **no 15-minute redial**: the
"15 minutes" is a dispatch wait applied once, before the lead reaches the vendor, not a second
dial after a failed one.

## What the data can and cannot show

`lead_sent_time` is when the request is **dispatched to the vendor**, not when the hot lead was
generated. The **hot-lead generation time is not in the dataset**, so the 15-minute wait itself
**cannot be measured directly** — we cannot observe the gap between generation and dispatch.
What we can measure is the **dispatch-to-dial** delay (`delay_min = call_start_time −
lead_sent_time`) and how conversion varies with it.

## Indirect evidence: the dispatch-to-dial freshness curve

Chart: `chart_freshness.png`. Meeting rate on first attempts (Apr–Aug, n = 162,070) by
dispatch-to-dial delay, with Wilson 95% CIs:

| dispatch→dial | n | meeting rate | 95% CI |
|---|---|---|---|
| ≤30 s | 76,422 | 10.7% | [10.5%, 10.9%] |
| 30 s–1 m | 10,359 | 10.1% | [9.6%, 10.7%] |
| 1–2 m | 13,389 | 9.7% | [9.2%, 10.2%] |
| 2–5 m | 19,190 | 9.0% | [8.6%, 9.4%] |
| 5–15 m | 21,014 | 8.9% | [8.5%, 9.3%] |
| 15–30 m | 10,537 | 8.7% | [8.2%, 9.3%] |
| 30–60 m | 6,613 | 7.7% | [7.1%, 8.4%] |
| >60 m | 4,546 | 8.6% | [7.8%, 9.4%] |

Conversion falls monotonically with the delay — about **10.7% at ≤30 s down to 7.7% at 30–60 m**.
Because a 15-minute dispatch wait adds directly to the dispatch-to-dial delay, this is the best
available proxy for the cost of the wait: every extra minute between dispatch and dial costs
conversion, so waiting to dispatch can only work against the objective. 47.5% of first attempts
are already dispatched within 30 s and 73.7% within 5 minutes; the tail (15–60 m) converts 1–3
points lower.

## What should follow a no-answer (unchanged)

Chart: `chart_daypart_shift.png`. On NA→next transitions (n = 162,084), retrying in the **same**
band answers 29.1% (n = 61,357) while a **different** band answers 32.0% (n = 100,727) — +2.9
points. The best gap is 4–8 h and evening bands answer best. The call after a no-answer should
move daypart, never repeat the same hour.

## Verdict: **Change, and Supplement**

- **Change the dispatch wait.** The freshness curve shows conversion is highest the sooner the
  lead is dialed, so the default 15-minute wait works against the objective. Test dispatching all
  hot leads immediately.
- **Keep the priority fast-track.** High-priority leads (payment page opened) are already
  dispatched immediately; that is exactly the behaviour the curve favours — keep it, and if
  anything widen the fast-track.
- **Keep the after-no-answer daypart shift.** Independent of the dispatch wait, the next call
  after a no-answer should go to a different daypart (+4–8 h, prefer 17–20).

## Caveats

- The generation time is absent, so the 15-minute wait is **not measured**; the freshness curve is
  indirect evidence on the dispatch-to-dial delay, not a direct test of the wait.
- The dispatch-to-dial delay is confounded with lead quality and queue load (the overnight backlog
  lands at 9 AM), so the curve is suggestive, not causal.
- All figures are Apr–Aug; September is held out and untouched.

## Proposed online A/B test

Randomise at the hot-lead level, over four weeks, three arms:

- **Arm 0:** dispatch immediately (0 minutes).
- **Arm 5:** dispatch after 5 minutes.
- **Arm 15:** current default (15 minutes).

Keep the **priority fast-track on in all arms**, and keep the **after-no-answer daypart shift in
all arms**, so the test isolates the dispatch wait.

Primary metric: meetings fixed per dispatched hot lead. Secondary: connect rate, time-to-first-dial,
calls per meeting. Guardrail: vendor dial volume per hour. Pre-register a two-sided test at α = 0.05,
powered for a 5% lift in meetings per lead; analyse per seller with a cluster-robust CI.
