# skills.md: build journey and tools

## Tools we used
| Tool | What we used it for |
|---|---|
| **Sarvam Cowork** | Agents BTC Analyst and Callback Extractor, four custom skills, the knowledge base "BTC-Context", and Python execution for the whole modelling pipeline |
| **Sarvam Voice Agents** | The "BTC Smart Callback" voice agent: Hinglish prompt, Bulbul v4 voice (Ishita), input variables, post-call output variables, call goal, phone test calls |
| **Sarvam LLM** | Post-call extraction (callback time, hint, wrong number, "don't call"), and running the Cowork agents |
| **Claude Code** | Exploratory analysis, planning, reviewing agent outputs, the Sept full-system analysis, checking a teammate's alternative model, slides |
| **Python** | pandas, NumPy, LightGBM, scikit-learn, SciPy, matplotlib, pyarrow |

### Cowork skills we wrote (`sarvam/cowork/skills/`)
- `btc-data-rules`: data cleaning, definitions, point-in-time rules, and the April–July / August / September split.
- `btc-backtest`: how every policy is evaluated: replay, SNIPS, doubly-robust, seller bootstrap.
- `callback-time-extraction`: turns Hinglish timing talk into a callback time ("kal shaam 6 baje", "Diwali ke baad").
- `btc-callback-scheduler`: combines the seller's words with the model windows into the next call time.

## Journey
### Day 1: understand the data before modelling
1. **Read the brief and scoring.** The target is meetings fixed, and the 15-minute rule needs a verdict.
2. **Exploratory analysis of 530,748 calls** (`src/eda/`). We found:
   - the "not interested" and "general" labels swing week to week, so we use meetings fixed as the target;
   - callback timestamps are system defaults, so they're never used as labels;
   - the raw hour table is distorted by retries and the overnight backlog, so hours are measured on fresh first calls.
3. **Key test: does the best hour depend on seller type?** We fitted segment-specific hour patterns for 7 traits and scored them on September. None helped. The hour pattern is shared, and seller type changes only the level. **Design decision:** a policy-first system with a shrinkage timing model, not one hour model per seller.
4. **Cowork setup.**
   - Wrote the knowledge base (problem brief, data dictionary, verified findings, solution design, master context).
   - Created the skills; we had to fix their SKILL.md frontmatter to start with `---` before they would upload.
   - Built the BTC Analyst agent.

### Day 1 night to Day 2 morning: the pipeline in Cowork (six steps, P0–P6)
- **P0 to P1:** load and QA. Every headline number reproduced: 526,718 dialed calls, 51.2% connect rate, 5.9% meetings.
- **P2:** we built the evaluator first.
  - **Lesson:** the first lift estimate (+12%) assumed a lead moved to a later hour converts as if it were fresh. We added the cost of waiting, never moved fresh leads, and the honest figure became +3.5% for the plain best-hour rule.
- **P3:** hierarchical timing model, giving windows for 149,363 sellers. Its lift over the plain best-hour rule was confirmed with a paired bootstrap.
- **P4:** retry decision table from 162,373 missed-call follow-ups.
- **P5:** LightGBM stop model (AUC 0.744, calibrated). A bug that merged episodes across sellers was caught and fixed.
- **P6:**
  - full policy with ±20% hourly capacity limits;
  - waiting periods implemented as *deferrals* rather than deletions;
  - the trade-off curve between meetings and calls;
  - rolling monthly folds.

  **Result:** +7.2% meetings, or 14.1 calls per meeting.

### Day 2: the Sarvam voice and LLM layer
1. **Callback Extractor in Cowork,** run on 2,000 call-later summaries.
   - **Lesson:** the agent's first extractor was keyword-based and marked 37% of rows "no signal". We checked it against careful reading of 250 rows, fixed 5 patterns, then measured honestly on 100 untouched rows: **93%** agreement.
   - It also found that **19%** of call-later calls were wrong numbers.
2. **Voice agent "BTC Smart Callback" in Voice Agents.**
   - **Lesson 1, debugged with diagnostic output variables:** post-call extraction **cannot read input variables**; the extractor was inventing hours (17:00). **Fix:** the agent now *says* the model's window in the call, so the extractor reads it from the transcript.
   - **Lesson 2:** the knowledge-base lookup searches by meaning, so it couldn't fetch an exact seller's window from a 20,000-line file. We moved to exact input variables, which a campaign cohort CSV fills per seller.
   - **Business clarification:** a callback is a time for the next phone call, not a meeting, so the call goal became "best time secured". We also stopped objecting to late-night times.
3. **September full-system test** on 67,008 summaries (`src/llm_layer/`).
   - **Finding:** calling *before* the seller's stated time is what hurts (4.5% vs 6.8% meetings).
   - **Suppression:** wrong number, "don't call" and hard "no" save about 2,000 calls a month and lose 26 meetings.
   - **Combined result:** +7.4% meetings.
   - **Reader accuracy:** 91% against 200 human-labelled calls.
4. **15-minute rule.** Our first reading (a vendor redial) was wrong. The team clarified it's a 15-minute wait before dispatch, and we rewrote the verdict.

## What we'd do next
- 4-week seller-level A/B test of the retry rules and the 0/5/15-minute dispatch.
- A telephony connection plus a campaign cohort generated from the model each month (prompt P7), so windows reach every call automatically.
- Monthly retraining. The models need at least 2 months of history.
