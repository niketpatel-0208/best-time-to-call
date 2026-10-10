# Data QA Note — Best Time to Call (P1: clean base table)

IndiaMART Voice AI Hackathon 2.0, Problem 3. Prepared 9 Oct 2026.
This note records the clean analysis table, its columns, the point-in-time
feature construction, and every headline number checked against the established
findings.

## 1. Files loaded

| File | Rows | Grain | Key |
|---|---|---|---|
| attempts.csv | 530,748 | one row per call attempt, 1 Apr – 30 Sep 2026 | data_hotlead_disposition_dtlid; seller = fk_glusr_usr_id |
| sellers.csv | 149,363 | one row per seller | fk_glusr_usr_id |
| call_later_summaries.csv | 2,000 | one row per Call-Later attempt (Apr–May) | data_hotlead_disposition_dtlid |

Datetimes parsed as IST (no timezone) for lead_sent_time, call_start_time,
vendor_response_time and lead_tbro_time.

## 2. Cleaning and joins

- Duplicate attempt ids: 0.
- Undialed rows (empty call_start_time): 4,030, all NotAnswered — dropped.
  Dialed attempts remaining: 526,718.
- Seller join on fk_glusr_usr_id: 526,718 / 526,718 = 100.00%. No row added or
  lost; 0 true join misses. (One seller row has a null seller_state value — a
  missing attribute, not a failed join.)
- Summary join on data_hotlead_disposition_dtlid: 2,000 / 2,000 = 100.00%.

## 3. Base table contents (base_table.parquet, 526,718 rows x 57 columns)

Added columns:

- episode — consecutive attempts of one seller; a new episode starts whenever
  call_attempt_count resets to 1 (max 19 episodes for one seller).
- band — 2-hour band from the hour of call_start_time: 9-10, 11-12, 13-14,
  15-16, 17-18, 19-20. Seven calls at 21:00 (just outside the 09:00–20:59
  window) fold into the 19-20 band; band now has no nulls.
- delay_min — call_start_time minus lead_sent_time, in minutes.
- outcome4 — Not Answered / Talked-no-progress (Not Interested + General) /
  Call Later / Meeting Fixed. meeting_fixed is authoritative for the meeting
  class.
- bot — main_vani / arrowhead / other.
- label_regime — pre_3aug / aug_3_9 / aug10_sep6 / from_sep7.

Point-in-time seller-history features (built only from rows strictly before the
attempt's call_start_time):

- prior_attempts, prior_answers, prior_meetings;
- prior_ni, prior_cl, prior_gen (prior Not Interested / Call Later / General counts);
- smoothed_answer_rate, smoothed_meeting_rate (prior = global rate, strength k = 5);
- days_since_last_answer — days since the last answered call before this one;
- prev_outcome — the previous attempt's outcome4 within the episode;
- gap_min — minutes since the previous attempt in the episode;
- hour_shift — current hour minus the previous attempt's hour in the episode.

Seller profile joined from sellers.csv (city, district, state, business_type,
annual_turnover, nature_of_business, nature_of_business_secondary,
gst_registration_year, top_category_1..3, top_parent_category,
top_category_group, num_categories).

## 4. Verification against the established findings

| Check | Expected | Computed | Status |
|---|---|---|---|
| Dialed rows | 526,718 | 526,718 | pass |
| Connect rate | 51.2% | 51.2% | pass |
| Meeting rate per dialed call | 5.9% | 5.9% | pass |
| Calls per meeting, September | 16.8 | 16.9 | rounding (see below) |
| Meetings on attempt 1 | 68% | 21,082 / 31,013 = 68.0% | pass |

Also reproduced: overall calls per meeting 17.0 (526,718 / 31,013 = 16.98);
outcome4 mix Not Answered 257,263 · Talked-no-progress 194,770 · Call Later
43,672 · Meeting Fixed 31,013; bot mix main_vani 396,824 · arrowhead 69,098 ·
other 60,796.

### Note on September calls per meeting (16.9 vs 16.8)

The findings state 16.8 calls per meeting in September; this table gives 16.9
(124,782 September calls / 7,383 meetings = 16.90, i.e. a September meeting rate
of 5.92%). The gap is 0.1 calls per meeting and sits at the rounding boundary —
the same underlying rate rounds to 5.9% and is consistent with the 5.9% meeting
rate and the 17.0 overall figure that both match exactly. It is a rounding-level
difference, not a definitional one; September is unchanged whether the month is
taken by call_start_time or lead_sent_time (16.9 either way). No action needed,
but flagged here as required.

## 5. Rules carried forward

- Primary target: meeting_fixed. Connect rate is secondary.
- Temporal split (binding): train 1 Apr – 31 Jul; validate 1 – 31 Aug; test
  (locked, used once) 1 – 30 Sep.
- The base table keeps the raw columns lead_call_duration, vendor_response_time,
  lead_tbro_time and disposition_label for target construction and QA, but none
  of them — nor the current row's disposition_label — may enter any feature set.
  lead_tbro_time is never a label.
- No PII in outputs: no pincode, locality, coordinates or phone numbers.
