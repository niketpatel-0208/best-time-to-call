---
name: btc-backtest
description: How to evaluate any seller-calling policy against the current schedule on held-out September data, using stratified replay, SNIPS and doubly-robust estimators with seller-clustered bootstrap CIs. Use whenever comparing policies or reporting lift in meetings, connect rate or calls per meeting.
---

# Skill: Best-Time-to-Call policy backtest

Use this skill to compare any calling policy against the current schedule on held-out data.

## Inputs
- A policy function: (seller context at decision time) → recommended band (plus a day offset for retries and follow-ups).
- The test window: Sep 2026, unless told otherwise. Only use episodes that start inside the window.

## Estimators (report all three)
1. **Stratified replay**
   - Strata are state_of_seller (previous outcome) × attempt number (1, 2, 3, 4+) × bucket group.
   - In each stratum, compare the meeting rate of logged calls whose band = the policy's band with the rate of the rest.
   - Pool the strata with CMH / inverse-variance weights.
   - Lift = pooled(match) / pooled(no-match) − 1.
2. **SNIPS**
   - Fit a propensity model P(logged band | context) on training months: multinomial, calibrated.
   - Weight = 1[logged band == policy band] / p(logged band). Clip at the 99th percentile.
   - V = Σ w·y / Σ w. Report the effective sample size (Σw)² / Σw².
3. **Doubly robust**
   - V = mean( q̂(x, π(x)) + w·(y − q̂(x, logged band)) ), where q̂ is the outcome model.

## Metrics per policy
- Meeting rate per call.
- Connect rate.
- Calls per meeting.
- Meetings retained vs. calls spent: the cap curve over 1–8 attempts.

Run each metric for the current schedule as well, and report:
- the lift with a 95% CI from 1,000 bootstrap resamples of sellers;
- splits by new vs. repeat seller, bot, and bucket.

## Overlap check
Before trusting SNIPS or DR, show the share of test traffic in each band per stratum. Mark any stratum where a band has under 2% of traffic as "low support".

## Report format
One table per metric:

| policy | value | lift vs. current | 95% CI | n | ESS |
|---|---|---|---|---|---|

Follow it with three bullet points of plain-language interpretation.
