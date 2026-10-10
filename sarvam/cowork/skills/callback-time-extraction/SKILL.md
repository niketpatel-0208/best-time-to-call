---
name: callback-time-extraction
description: Read a seller call transcript or summary (Hindi, English or Hinglish) and extract any indication of when the seller can be called again - explicit times ("kal shaam 6 baje") and implicit hints ("Diwali ke baad", "is hafte bahar hoon", "dukaan band hone ke baad") - as JSON with confidence and evidence. Use whenever a call transcript or summary needs a callback time, a do-not-call check, or input for the callback scheduler.
---

# Skill: extract callback timing from a seller conversation

## Input
- `text`: the full call transcript or call summary (any of Hindi, English, Hinglish, Devanagari or Roman script)
- `call_time`: when the call happened (IST)

## Output (JSON only, one object per call)
```json
{"has_time_cue": true,
 "cue_strength": "explicit|implicit|none",
 "cue_type": "clock|relative_minutes|relative_hours|relative_days|tomorrow|weekday|date|daypart|event|unavailable_period|vague|none",
 "callback_at": "2026-10-11T17:55:00+05:30",
 "window_start": null, "window_end": null,
 "earliest_date": null,
 "reason": "busy|travelling|unwell|festival|season|meeting|other|none",
 "do_not_call": false,
 "interest_signal": "positive|neutral|negative",
 "evidence": "<exact phrase from the text>",
 "confidence": 0.9}
```

## Explicit cues (`cue_strength` = explicit; fill `callback_at`)
- `callback_at` = the stated time − 5 minutes. Calling slightly early converts best; calling late hurts.
- "das minute / 10 min baad" → call_time + 10 min. "ek ghante baad" → + 60 min. "do ghante baad" → + 120 min.
- Clock time ("6 baje", "4:30") → that time today. If it has already passed, tomorrow.
- "kal" → tomorrow. "parson" → the day after tomorrow. A weekday name → the next occurrence of that day. If no clock time is given, use 11:00.
- A date ("30 tareekh", "5 November") → that date (this month, or next month if it has passed), 11:00 unless a time is stated.

## Implicit hints (`cue_strength` = implicit; fill `window_start`/`window_end` and/or `earliest_date`)
| Hint (examples) | Interpretation |
|---|---|
| subah / morning, "dukaan khulne pe" | window 10:00–12:00 |
| dopahar / afternoon, "lunch ke baad" | window 14:00–16:00 |
| shaam / evening, "kaam ke baad", "dukaan band hone ke baad" | window 17:00–20:00 |
| raat / night | window 19:00–20:30 |
| "is hafte bahar hoon", "travel kar raha hoon" (no end date) | earliest_date = next Monday |
| "kuch din baad", "2-4 din baad" | earliest_date = call_time + lower bound in days (default 3) |
| "5-10 din tak available nahi", unwell for N days | earliest_date = call_time + lower bound |
| unwell, no duration | earliest_date = call_time + 2 days |
| "next week" | earliest_date = next Monday |
| "month end ke baad", "agle mahine" | earliest_date = 1st of next month |
| festival ("Navratri / Dussehra / Diwali ke baad") | earliest_date = first working day after the festival (calendar below); confidence ≤ 0.6 |
| "season chal raha hai", "abhi bahut busy hoon" (no time) | earliest_date = call_time + 14 days; confidence ≤ 0.5 |

A daypart combined with a day ("kal shaam") gives a window on that day. Treat it as implicit unless a clock time is given.

**Festival calendar 2026** (team to verify before the demo): Navratri 11–19 Oct, Dussehra 20 Oct, Diwali 8 Nov. For an unknown festival, use call_time + 7 days with confidence ≤ 0.5.

## Vague or none
- "baad mein", "free hoke", "jab time ho", with nothing else → `cue_type` "vague", `cue_strength` "none", all times null. The scheduler then uses the model's window.
- No callback talk at all → `cue_type` "none".

## Other fields
- `do_not_call` = true only if the seller explicitly asks never to be called again ("dobara call mat karna", "number hata do").
- `interest_signal`: positive (asks questions, agrees to talk later), negative (clearly not interested), otherwise neutral.
- `reason`: why the seller couldn't talk now.

## Rules
- Calls are only allowed 09:00–21:00. Move any time or window outside this to the nearest allowed time.
- Never invent a time the text does not support. `evidence` must quote the exact source phrase.
- Confidence: explicit clock or relative time 0.85–0.95; day without time 0.7–0.8; daypart or other implicit hint 0.5–0.7; festival or season 0.3–0.6.
- If the transcript contains a time the agent proposed and the seller agreed to ("haan theek hai"), that counts as explicit.
