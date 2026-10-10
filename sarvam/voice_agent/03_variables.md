# Agent variables

## A. Input variables (filled per call from the campaign cohort, built from the model output)
Date and time come from Sarvam's built-in variables `@current_day`, `@current_date` and `@current_time`.

| Name | What it holds | Example (demo seller) |
|---|---|---|
| seller_id | Seller reference | `s_xxxxxxxxxx` |
| seller_name | Seller's first name | Imran |
| company_name | Company name | Imran Paints & Chemicals |
| top_category | Top product category | Paint Chemicals |
| seller_city | City | New Delhi |
| rec_window | Best window from the timing model, in Hinglish | subah 11 se 12 baje ke beech |
| backup_window | Backup window from the timing model, in Hinglish | shaam 5 se 6 baje ke beech |

## B. Post-call variables: the callback-time layer
Filled by Sarvam's LLM reading the full transcript after the call ends. Call goal: `best_time_secured = true`.

**call_outcome**: enum
Options: `Meeting Fixed`, `Call Later`, `Not Interested`, `General`, `Do Not Call`
```
Classify the outcome of this call as exactly one of: Meeting Fixed (seller agreed to meet an IndiaMART executive in person, at a day/time), Call Later (seller agreed to or asked for a later phone call, gave a time or hint, or was busy; a callback is NOT a meeting), Not Interested, General (talked but no outcome), Do Not Call (seller asked never to be called again).
```

**callback_json**: string (the full extraction; the same format the Cowork Callback Extractor produces)
```
Read the whole transcript. Use the call's start date and time as the call time. Return ONLY one JSON object with these keys:
has_time_cue (true/false); cue_strength (explicit|implicit|none); cue_type (clock|relative_minutes|relative_hours|relative_days|tomorrow|weekday|date|daypart|event|unavailable_period|vague|none); callback_at (ISO 8601 +05:30, explicit times only, set to the stated time minus 5 minutes, else null); window_start and window_end (ISO, for daypart hints: subah 10:00-12:00, dopahar 14:00-16:00, shaam 17:00-20:00, raat 19:00-20:30, else null); earliest_date (YYYY-MM-DD for hints: 'is hafte bahar hoon' or 'next week' = next Monday, 'kuch din' = +3 days, unwell with no duration = +2 days, 'Diwali ke baad' = 2026-11-09, 'Dussehra ke baad' = 2026-10-21, 'season chal raha hai' = +14 days, else null); reason (busy|travelling|unwell|festival|season|meeting|other|none); do_not_call (true only if the seller asked never to be called again); interest_signal (positive|neutral|negative); evidence (the seller's exact words); confidence (0-1: explicit 0.85-0.95, day only 0.7-0.8, daypart or hint 0.5-0.7, festival or season 0.3-0.6).
Being busy or saying 'baad mein' / 'later' / 'free hoke' with no time or hint is NOT a hint: then has_time_cue=false, cue_strength=none, cue_type=vague, and all times null. A time the agent proposed and the seller agreed to counts as explicit. Never invent a time the transcript does not support.
```

**callback_at**: string (the FINAL next-call time, read from the conversation)
Note: after-call prompts cannot read input variables (confirmed by a debug test). They see the transcript and the call's start time. The agent therefore always SAYS the timing model's window in the call, and this prompt reads it from the transcript.
```
Return ONLY an ISO 8601 timestamp with +05:30, or an empty string. Use the call's start date and time as "now". Follow these steps in order and stop at the first that applies.
Step 1. The seller asked never to be called again, or agreed to meet an IndiaMART executive in person -> return an empty string.
Step 2. The seller stated a specific time, or agreed to a specific time or window the agent offered -> return that time (for a window, its start) minus 5 minutes if the seller named it, or exactly the window start if the agent offered it and the seller agreed.
Step 3. The seller gave only their own hint about when (a part of the day, a day, 'next week' = next Monday, 'kuch din' = +3 days, unwell = +2 days, 'Diwali ke baad' = 2026-11-09) -> on the hinted day, use the start of the window the agent offered in the call if it fits the hint; otherwise the start of the hinted part of the day (subah 10:00, dopahar 14:00, shaam 17:00, raat 19:00); otherwise 11:00.
Step 4. Otherwise (busy with no time, vague, hung up, not interested, talked with no outcome): take the LAST time window the agent offered or said it would try (for example the agent said "aaj shaam 7 se 8 ke beech" -> today 19:00) that the seller did not reject, and return its start. Do not invent an hour. If the agent never said any window, return an empty string.
```

**callback_window**: string (one readable line for the sales team, shown on the call screen)
```
One line in English describing callback_at for the sales team: day, date and time, then who decided and why. If it came from a window the agent offered (the timing model's window), write the full window as spoken in the call, in 12-hour form, e.g. 'Sat 10 Oct, 7-8 PM - timing model's best window (seller busy, gave no time)' or 'Sun 11 Oct, 11 AM-12 PM - timing model's backup window (seller declined 7-8 PM)'. If the seller decided, e.g. 'Sun 11 Oct, 5:55 PM - seller asked for 6 PM'. If callback_at is empty because the seller hung up before any window was said, write 'Use the timing model's best window for this seller (shown as Rec window 24h) - seller hung up before a time was offered'. Write 'Do not call - seller asked' or 'No bot call - in-person meeting agreed' when that applies.
```

**cue_strength**: enum
Options: `explicit`, `implicit`, `none`
```
explicit if the seller stated or agreed to a specific time or day; implicit only if they gave a real hint about WHEN (a part of the day like 'shaam ko', a day, 'after Diwali', 'out this week', unwell for some days); none if they only said they are busy or to call later with no time or hint ('baad mein', 'abhi busy hoon', 'traffic mein hoon', 'later'), or no callback was discussed.
```

**next_call_source**: enum
Options: `seller_stated`, `seller_hint`, `model_window`, `retry_policy`, `no_call`
```
Which rule decided the next call: seller_stated (seller gave or agreed a specific time), seller_hint (seller gave only a hint), model_window (seller was busy with no time or hint, so the timing model's window was used), retry_policy (not interested, or talked with no outcome), no_call (do not call, or an executive meeting was agreed).
```

**best_time_secured**: boolean if offered, otherwise enum `true` / `false` (this is the CALL GOAL)
```
true if the call ends with a usable next step: either a next-call time or window was set for this seller (from the seller's own words, a hint, or the timing model's window), or the seller agreed to meet an IndiaMART executive in person, or the seller was busy or hung up without refusing future calls (the timing model's best window then applies automatically). false only if the seller asked never to be called again, or no next call can be set.
```

**do_not_call**: enum
Options: `true`, `false`
```
true only if the seller explicitly asked never to be called again, otherwise false.
```

## Band → Hinglish (used to fill rec_window / backup_window)
| Band | Hinglish |
|---|---|
| 9-10 | subah 9 se 10 baje ke beech |
| 11-12 | subah 11 se 12 baje ke beech |
| 13-14 | dopahar 1 se 2 baje ke beech |
| 15-16 | dopahar 3 se 4 baje ke beech |
| 17-18 | shaam 5 se 6 baje ke beech |
| 19-20 | shaam 7 se 8 baje ke beech |

Prefix with "aaj", "kal" or the weekday, depending on the call date.
