# Code

## `model/`: the timing model, retry rules, stop model and evaluation
These scripts were written and run by the **BTC Analyst agent in Sarvam Cowork** (the six steps are described in `skills.md`). The Cowork sandbox reads inputs from `/scratch/input/` and writes to `/scratch/work/`. To run locally, create those two folders, or change the path constants at the top of each file.

Run them in this order:
| Step | Script | Produces |
|---|---|---|
| 1 | `qa.py`, `features.py` | Data QA; point-in-time feature table (`base_table.parquet`) |
| 2 | `evaluator.py` | Shared policy evaluator: replay, SNIPS, doubly-robust, cost of waiting, bootstrap CIs |
| 3 | `timing_model.py` | Hierarchical Bayesian windows → `seller_windows.csv` |
| 4 | `retry_policy.py` | Retry decision table → `retry_policy.json`; 15-minute rule evidence |
| 5 | `priority_model.py` | LightGBM stop model, cap curve and waiting periods → `model_card.md` |
| 6 | `backtest.py`, `frontier.py`, `folds.py` | Full policy under capacity limits, the meetings-vs-calls trade-off curve, rolling monthly folds → `backtest_report.md` |

LightGBM needs OpenMP (`libgomp`). In Cowork the scripts point `LD_LIBRARY_PATH` at a bundled copy; locally, `pip install lightgbm` is enough.

## `llm_layer/`: the Sarvam LLM layer, evaluated on September
| Script | Purpose |
|---|---|
| `extract.py` | The `callback-time-extraction` skill as rules: explicit time, hint, wrong number, "don't call" |
| `run_extract.py` | Runs the extraction over the September summaries |
| `check2.py` | Accuracy against 200 human labels; meeting rate by when the next call happened relative to the seller's time |
| `impact.py`, `final.py` | Full-system September result: model + LLM timing + suppression |
| `intent.py`, `intent2.py` | What each kind of summary means for the next call, and the suppression rules |

These were run locally with arguments `<project root> <work folder>`; see each script's `sys.argv`. They need the internal September summary file, which is not included.

## `eda/`
Exploratory analysis (`eda1–7.py`), the first feasibility model (`model0.py`), and `retry_nums.py`, which recomputes the retry-rule numbers on the slides.

```bash
pip install -r requirements.txt
```
