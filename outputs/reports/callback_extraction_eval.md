# Callback-time extraction — evaluation (B1)

## Method

No human labels exist for the 2,000 Call-Later summaries, so the rule-based
extractor was validated against an independent **LLM-judgement reading** of a
sample. Each sampled summary was read and labelled by hand, applying the
`callback-time-extraction` skill by judgement rather than by regex, and the
result was compared with the extractor's output.

- Sample: **200 random rows** (seed 7) from all 2,000, plus **50 random rows**
  (seed 7) drawn specifically from the rows the extractor had called `none`.
  250 rows in total (5 overlap between the two draws).
- Per-row judgement recorded: `has_time_cue`, `cue_strength`, `cue_type`, and
  the target (`callback_at` for explicit cues; window / `earliest_date` for
  implicit hints).
- Agreement was measured on: `has_time_cue` (boolean), `cue_strength`,
  `cue_type`, and the target **same-day** (and, for explicit cues, within
  ±60 minutes).

Because the reading and the extractor are both imperfect, this is an
inter-annotator style check, not ground truth. It is strong enough to expose
*systematic* misses and weak enough that a handful of genuine judgement
disagreements remain.

## Agreement before and after the fix

| Metric | Before fix | After fix |
|---|---|---|
| `has_time_cue` | 230/250 = **92.0%** | 250/250 = **100.0%** |
| `cue_strength` | 224/250 = **89.6%** | 249/250 = **99.6%** |
| `cue_type` | 177/250 = **70.8%** | 242/250 = **96.8%** |
| target — same day (both cued) | 79/88 = **89.8%** | 102/103 = **99.0%** |
| target — day + window | 78/88 = **88.6%** | 101/103 = **98.1%** |

`cue_type` was the weakest metric before the fix and improved the most
(+26.0 pp): most of that came from rows the extractor had wrongly left as
`none` or `vague` instead of a real cue type.

## Held-out agreement (fresh sample, not used for tuning)

The "after" numbers above were measured on the same rows used to design the
fixes, so they are optimistic. To get an honest estimate, a **fresh 100-row
sample** was drawn with seed 11, excluding all 250 earlier rows, read by
judgement, and compared with the same fixed extractor (no further tuning).

| Metric | Design set (250) | Held-out (100) |
|---|---|---|
| `has_time_cue` | 100.0% | **93.0%** |
| `cue_strength` | 99.6% | **92.0%** |
| `cue_type` | 96.8% | **84.0%** |
| target — same day (both cued) | 99.0% (n=103) | **91.1%** (n=56) |
| target — day + window | 98.1% | **87.5%** |

The held-out rate is 7–13 pp lower on every metric — the expected shrinkage.
The main held-out disagreements were:
- **Past-reference dates read as future cues (2 rows).** "a previous
  conversation on April 26th" and "previously requested a callback after a
  month" were taken as forward-looking dates (one produced a 2027 date). This is
  the most damaging class and the clearest thing to fix next.
- **A weekday inside an unavailability / declined context (2 rows)** read as a
  callback day ("unavailable for a meeting on Monday and on Sunday").
- **Day-off-by-N on clock + relative combos (3 rows)** where a clock time and a
  relative offset appear together.
- **vague-vs-none / vague-vs-implicit boundary (several rows)** and a couple of
  same-day `cue_type` nuances.

These are left unfixed deliberately: tuning on the held-out set would destroy
its value as a check.

## Disagreement patterns found (and fixed)

1. **Vague deferrals read as "none" (largest pattern, 42 rows).** The first
   vague detector only matched a short list of phrases. Sellers who said they
   were *busy / not available / would call back / would inform later* — with no
   time — were left as `none`.
   *Example:* "The lead stated that he is busy and asked to call him later."
   *Fix:* the vague vocabulary was widened to any deferral of the conversation
   (busy at the moment, could not talk, will revert, when free, try later,
   asked not to call, did not provide a time, …). → down to 1 row.

2. **Agent-proposed times read as seller cues (8 rows).** When the agent asked
   *"should I call back in one to two hours?"* or *"offered to call back in an
   hour or two"* and the seller did not commit, the extractor still took it as
   an explicit cue.
   *Example:* "The agent offered to call back in an hour or two" (lead never
   agreed).
   *Fix:* a local-context guard ignores a relative time that sits inside an
   agent question/proposal unless the same sentence records agreement
   ("agreed", "confirmed", "theek hai"). → 0 rows.

3. **Day words inside an unavailability or a closure (5 rows).** "not available
   tomorrow", "out of station the next day", "Monday is closed", "the executive
   was supposed to visit on Monday" were read as callback days.
   *Example:* "He is not available tomorrow and will provide a tentative time
   later." → wrongly `tomorrow`.
   *Fix:* a day anchor is skipped when the day is framed as unavailability
   (not available / out / away / difficult) or as a closure/visit, and a
   weekday immediately followed by "closed/off/holiday" is skipped. → 0 rows.

4. **Month-level periods missed (5 rows).** "second week of July", "last week
   of June", "available in June", "call back in December", "10 May" produced
   `none`.
   *Fix:* added a month-period detector (first/second/last week of <month>,
   "in/after <month>", "after this month") and a day+month date pattern.
   → 0 rows.

5. **Travel / "out" hints: reading vs. rule.** A few rows where the seller said
   they were *out / travelling* and asked to be called later were labelled
   `vague` in the first reading but `implicit / unavailable_period` by the
   extractor. The skill is explicit that "travelling, no end date" is an
   implicit hint (earliest_date = next Monday), so the reading was corrected
   to match the skill. This is a labelling convention, not an extractor bug.

**Residual disagreements after the fix (9 rows, all minor):**
- `cue_type` nuance where both are explicit and the date agrees, e.g. "on the
  18th between 11 AM and 5 PM" read as `date` vs `clock`, or "next Monday
  between 11 AM and 5 PM" read as `weekday` vs `clock` (3 rows).
- Word-ordinal "after the first of next month" — extractor returns the right
  date (1 May) but as `implicit` rather than `explicit` (1 row).
- The vague-vs-none boundary on rows where a *meeting* (not the call) was
  declined, or where the reply was a referral to another person, or the agent
  asked a question the seller did not answer (5 rows). The extractor is
  slightly more willing to call these `vague`; the reading left some as `none`.

## What the "none" rows actually contain

Every row is disposition *Call Later / Busy*, so a bare "no callback talk"
should be rare. After the fix there are **549 `none` rows (27.4%)**, and they
break down as:

| Category | Rows | Share of none |
|---|---|---|
| Wrong number / "not from that company" | 381 | 69.4% |
| No callback talk (meeting declined, info only, referral) | 157 | 28.6% |
| Referred to another contact/number | 9 | 1.6% |
| Account action (asked to delete the listing) | 2 | 0.4% |

So the dominant explanation is **misdialed / wrong-number calls**: the agent
reached someone who was not the seller, the call ended, and the system still
tagged it *Call Later*. These carry no callback intent at all and are correctly
`none`. The genuine remainder — about 157 rows — are calls where the seller
talked but declined a meeting or exchanged information without any timing
signal. The earlier version's much larger `none` bucket (737 rows) was an
artefact of the narrow vague detector, not of the data.

**Applied — the wrong-number calls are excluded from the callback population.**
The 381 wrong-number rows (and the 2 "remove my number" delete requests) are not
sellers and should never be scheduled for a callback; a further attempt is wasted
dialing and, for the "remove my number" cases, a compliance risk. In
`callback_schedule.csv` these rows now carry source **`wrong_number`** with no
`next_call_at`, and they are removed from the callback denominator in
`callback_impact.md` (the estimate is computed on the remaining 1,619 rows). Only
about **168 of the 549 `none` rows are real sellers** with no timing signal.

## Limitations

- **Reference is an LLM reading, not human labels.** Agreement numbers are an
  upper bound on true quality; the reading and the extractor share English-only
  assumptions.
- **Rule-based extractor.** It approximates the skill with regex. It handles the
  formulaic English summaries well but would generalise less reliably to raw
  Hindi/Hinglish transcripts (the intended Sarvam Voice Agent input), where the
  skill's daypart and festival vocabulary differs.
- **Festival dates are hard-coded** (Diwali 8 Nov 2026, Navratri/Dussehra in
  October) and, per the skill, should be verified before the demo.
- **Some cues are genuinely ambiguous** (agent-proposed times the seller never
  confirmed; "out" that may mean out of the office rather than out of town).
  These are recorded with low confidence rather than forced.
- **Sample size.** 250 rows gives roughly ±3–6 pp resolution on the agreement
  rates; the two smaller metrics (target day/window) rest on ~103 rows.
