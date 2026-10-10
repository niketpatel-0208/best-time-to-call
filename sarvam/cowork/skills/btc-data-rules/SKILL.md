---
name: btc-data-rules
description: Cleaning, definitions, point-in-time feature rules and the Apr-Jul / Aug / Sep time split for the IndiaMART Best-Time-to-Call call-attempt data. Use whenever loading, joining, feature-engineering or modelling attempts.csv, sellers.csv or call_later_summaries.csv.
---

# Skill: Best-Time-to-Call data rules

Use this skill whenever you load, clean, join, feature-engineer or model the Best-Time-to-Call data.

## Loading
- `attempts.csv`: parse `lead_sent_time`, `call_start_time`, `vendor_response_time` and `lead_tbro_time` as datetimes (IST, no timezone).
- Drop rows where `call_start_time` is empty (never dialed).
- `sellers.csv`: join on `fk_glusr_usr_id` (left join; it matches 100%).
- `call_later_summaries.csv`: join to attempts on `data_hotlead_disposition_dtlid`. It covers Call Later / Busy only, Apr–May only.

## Definitions
- answered = `lead_call_status == "Answered"`
- meeting = `meeting_fixed == 1` (do not use the text label for this)
- outcome4 = {Not Answered, Talked-no-progress (= Not Interested + General), Call Later, Meeting Fixed}
- hour = hour of `call_start_time`. 2-hour band = 9–10, 11–12, 13–14, 15–16, 17–18, 19–20.
- delay_min = `call_start_time − lead_sent_time`, in minutes
- episode = consecutive attempts of one seller, starting each time `call_attempt_count` resets to 1
- fresh first attempt = attempt 1 with delay_min ≤ 5. Use it for clean time-of-day effects.
- bot = main_vani / arrowhead / other
- label_regime = before 3 Aug / 3–9 Aug / 10 Aug–6 Sep / from 7 Sep

## Hard rules
1. **Point-in-time only.** A feature for attempt *t* may use rows strictly before `call_start_time` of *t*.
2. **Never use as a feature:** `lead_call_duration`, `vendor_response_time`, `lead_tbro_time` or `disposition_label` of the current row.
3. **Never use `lead_tbro_time` as a target.** It holds system defaults.
4. **Split by time, never at random.** Train: Apr–Jul. Validate: Aug. Test: Sep, locked and used once.
5. Bot and label_regime are controls. They are never a decision.
6. The primary target is meeting. Report connect rate as secondary.
7. Do not output pincode, locality, coordinates or phone numbers.
8. Every number you report states its n. Every policy comparison carries a 95% CI from a bootstrap that resamples sellers.
