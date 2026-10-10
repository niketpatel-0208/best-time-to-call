# Best Time to Call: IndiaMART Voice AI Hackathon 2.0, Problem 3

**For every seller, we decide when the voice bot should call again, and how to retry when nobody picks up, so that more calls end in a meeting with fewer calls wasted.**

Built on **Sarvam**: Cowork agents for the analysis, and a Voice Agent that captures each seller's own callback time.

| | |
|---|---|
| Voice agent | **BTC Smart Callback**, Sarvam Voice Agents, agent ID `BTC-Smart-C-a066330b-9321` |
| Build journey | [skills.md](skills.md) |

---

## Results: September 2026, a month the models never saw
| | Today | Our system, max-meetings setting | Our system, fewest-calls setting |
|---|---|---|---|
| Meetings / month | 7,383 | **7,926 (+7.4%)** | 7,527 (+2.0%) |
| Calls per meeting | 16.9 | 15.5 | **13.8** |
| Meetings per 1,000 calls | 59.2 | 64.4 | **72.4** |
| Calls / month | 124,782 | 122,987 | **103,909** |

- Timing model and call policy alone: **+7.2%** meetings (likely range +5.9% to +8.4%, from 1,000 resamples of sellers). The Sarvam LLM layer adds the rest and removes about 2,000 unwanted calls a month.
- **Consistent month after month:** +9.6% in July, +7.0% in August and +7.2% in September, each measured only on calls after its training period. The models refresh monthly from new calls.
- Details: [outputs/reports/backtest_report.md](outputs/reports/backtest_report.md) and [outputs/reports/full_system_backtest_sep.md](outputs/reports/full_system_backtest_sep.md).

## What we found before building
- **The best hour is the same for every kind of seller.** We tested 7 seller traits (state, turnover, lead type, category, business type, nature of business, GST year). Giving each type its own hour pattern added no accuracy on September. Type changes *how many* meetings, not *when*. For fresh first calls, meetings peak at 11 AM–12 PM (12.3 per 100) and 7–8 PM (13.0 per 100).
- **A seller's own history adds little:** pick-up at a seller's usual hour was 61.7%, vs 59.5% at other hours, and 61% of September's sellers had no history.
- **Raw hourly numbers mislead:** 11 AM looks like the worst hour across all calls (3%) only because failed retries pile up there. Fresh first calls at 11 AM book 12%.
- **Labels drift:** "Not interested" and "General" swapped week to week, so we optimise for **meetings fixed**, the outcome that stays stable.

So the lever is the **calling policy**: when to retry, when to stop, and listening to the seller.

## What we built
| Part | What it does | How |
|---|---|---|
| **Timing model** | Best and backup 2-hour window for all 149,363 sellers | Hierarchical Bayesian model: the shared hour curve × the seller's segment × the seller's own history, which counts only when there is enough of it (`src/model/timing_model.py`) |
| **Retry rules** | When to call again after each outcome | Learned from 162,373 missed-call follow-ups and all outcome-to-next-call gaps (`src/model/retry_policy.py`, `outputs/retry_policy.json`) |
| **Stop model** | Stops calling when a meeting becomes unlikely | LightGBM on 31 facts known before the call; AUC 0.744; the top 10% of calls convert at 17.7%, the bottom 10% at 0.9%; stops below a 1-in-80 chance (`src/model/priority_model.py`) |
| **Sarvam voice agent** | Offers the seller's window and confirms the seller's own time | `sarvam/voice_agent/` (settings, prompt, variables) |
| **Sarvam LLM layer** | Reads every transcript for a time, a hint, a wrong number or "don't call" | Post-call output variables in Voice Agents, plus the same rules in batch (`src/llm_layer/`); matches human reviewers on 91% of 200 calls |

**Decision order for every next call:** "don't call" or wrong number → stop; the seller named a time → call then; the seller gave a hint → the best window inside it; nothing said → the model's window; no answer → retry rules and the stop model.

### The retry rules
| If the last call… | Call again | What the data showed |
|---|---|---|
| wasn't answered | 4–8 h later, 3–9 PM | 46% picked up vs 37% at the current +1 h |
| failed, retry next day | 2–3 h later than the failed hour | 2.7% meetings vs 1.5% at the same hour |
| "call later", no time given | same day, within 6 h | 6.6% vs 4.4% next day |
| "call later", time given | at that time, never earlier | 6.8% vs 4.5% when called early |
| talked, no outcome | after 3–7 days | 5.5% vs 1.9% within 3 days |
| "not interested" | after 14+ days | 4.6% vs 1.6% within 3 days |
| 4 tries done | stop | 96% of meetings come in the first 4 tries |

### The 15-minute rule
See [outputs/reports/fifteen_minute_rule.md](outputs/reports/fifteen_minute_rule.md). The rule is a 15-minute wait before a hot lead is dispatched; high-intent leads skip it. Conversion falls with every minute between dispatch and dial: 10.7% within 30 seconds, 7.7% at 30–60 minutes. Our recommendation is **change and supplement**:
- dispatch every hot lead as early as possible instead of after a fixed 15-minute wait;
- keep the priority fast-track for high-intent leads;
- after a missed call, move the next attempt to a different part of the day (the retry rules above).

## How we validated
- **Split by time:** trained on April–July, tuned on August, tested **once** on September. Features use only information from before each call.
- **Policy value estimated off-policy:** stratified replay plus a doubly-robust estimator, with confidence intervals from 1,000 resamples of sellers.
- **Realism built in:** fresh leads are never delayed (the cost of waiting is priced), and no hour gets more than 20% above today's dialer load.
- **Rolling monthly folds:** June, July and August each tested on its own.
- **LLM layer:** hand-labelled check on 200 calls (91% agreement, 94% precision, 97% on wrong numbers), plus an earlier held-out check of 93% on April–May summaries.

## Cost to run
- **Timing model:** windows are precomputed and refreshed monthly. No per-call model cost.
- **LLM layer:** one read per answered call, about 69,000 a month at roughly 1,000 input and 150 output tokens each. On Sarvam 105B that is **₹1,700–2,800 a month (2–4 paise per call)** at list prices of ₹15–29.28 input and ₹60–73.2 output per million tokens. On Gemini 3.5 Flash-Lite ($0.30 / $2.50 per million tokens) it is **about $47 a month**.

## Repository layout
```
README.md · skills.md
src/
  model/       timing model, retry policy, stop model, evaluator, backtest, frontier, folds (run inside Sarvam Cowork)
  llm_layer/   callback-time extraction rules, September full-system impact, intent analysis
  eda/         exploratory analysis scripts and the retry-rule numbers
sarvam/
  voice_agent/ agent settings, prompt, input and output variables
  cowork/skills/ the 4 Cowork skills: callback-time extraction, callback scheduler, data rules, backtest
outputs/
  reports/     backtest, model card, 15-minute rule, LLM evaluation and impact, data QA
  charts/      PNG charts
  samples/     200-row sample of seller windows (seller IDs hashed), demo cohort (no phone numbers)
  retry_policy.json
data/          data dictionary; the data itself is internal and not included
```

## Reproducing
See [src/README.md](src/README.md). The raw data is internal and isn't in this repository.

## Data and privacy
No raw call data, seller-level outputs, phone numbers, recordings of real sellers or transcripts are included. The sample files use hashed seller references.
