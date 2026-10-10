---
name: btc-callback-scheduler
description: Decide the next call time for an IndiaMART seller by combining the callback-time-extraction JSON from the last call with the Best-Time-to-Call model outputs (seller_windows.csv bands and retry_policy.json). Seller-stated times win, hints narrow the window, and the model decides when there is no cue. Use whenever a next_call_at, a retry time or a callback schedule is needed.
---

# Skill: schedule the next call

## Inputs
- `extraction`: the JSON from the callback-time-extraction skill for the last call
- `call_time`: when the last call happened (IST)
- `call_outcome`: Meeting Fixed / Call Later / Not Interested / General / Not Answered / Do Not Call
- From `seller_windows.csv` for this seller: `rec_band`, `backup_band_1`, `backup_band_2`
- From `retry_policy.json`: the not-answered, General and Not-Interested rules and the attempt cap
- `callbacks_so_far`: Call-Later callbacks already made in this episode

Bands are 9-10, 11-12, 13-14, 15-16, 17-18, 19-20. A band slot starts at its first hour (for example 19-20 → 19:00).

## Decision rules (apply in this order)
1. **Do not call:** `extraction.do_not_call` is true, or the outcome is Do Not Call → no next call. `source` = "do_not_call".
2. **Meeting Fixed:** no bot call. `source` = "meeting_fixed".
3. **Explicit time:** `cue_strength` is explicit, `confidence` ≥ 0.7, and `callback_at` is set → `next_call_at` = `callback_at`, clamped to 09:00–20:55. `source` = "seller_stated".
4. **Implicit hint:**
   - **Window only:** pick the first of `rec_band`, `backup_band_1`, `backup_band_2` that overlaps the window on the hinted day, and use that band's start. If none overlaps, use `window_start`.
   - **`earliest_date` only:** use the first `rec_band` slot on or after that date.
   - **Both:** use the window on `earliest_date`.
   - `source` = "seller_hint".
5. **Call Later with no usable cue** (vague, none, or confidence below threshold) → call back the same day, 1–6 hours after `call_time`, at the first of `rec_band` / `backup_band_1` / `backup_band_2` that starts in that range. If none does, use tomorrow's `rec_band` slot. Never default to "tomorrow 12:00". `source` = "model_window".
6. **Not Answered:** follow `retry_policy.json`. The next call goes to a different daypart from the failed one: +4–8 h, or the next day in the model's best unused band. Cap at 4 attempts. `source` = "retry_policy".
7. **General:** +3–7 days at the `rec_band` slot. **Not Interested:** +14 days at the `rec_band` slot; two NIs in a row → suppress for 30 days. If the extraction has an implicit hint with a later `earliest_date`, use the later date. `source` = "retry_policy".
8. **Callback cap:** if `callbacks_so_far` ≥ 3, treat a further Call Later as General (rule 7).

## Constraints
- Calls only 09:00–21:00 IST.
- `next_call_at` is never earlier than `call_time` + 10 minutes.
- Fresh new hot leads from the seller are called immediately. This skill schedules follow-ups only.

## Output (one JSON object per call)
```json
{"next_call_at": "2026-10-11T17:55:00+05:30",
 "band": "17-18",
 "source": "seller_stated|seller_hint|model_window|retry_policy|do_not_call|meeting_fixed",
 "reason_code": "stated 'kal shaam 6 baje' (explicit, 0.9)",
 "fallback_used": false}
```
`reason_code` is one short plain-English line a sales manager would understand.
