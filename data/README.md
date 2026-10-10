# Data (not included)

The data is internal IndiaMART data and stays on the premises, so none of it is in this repository. To reproduce, place these files here:

| File | Rows | Content |
|---|---|---|
| `attempts.csv` | 530,748 | One row per call attempt, April–September 2026 (from `data_hotlead_disposition_dtl`) |
| `sellers.csv` | 149,363 | One row per seller: city, state, business type, turnover, categories, GST year. Drop pincode, locality and coordinates |
| `call_later_summaries.csv` | 2,000 | Call-later summaries, April–May |
| September call log with summaries | 124,782 | Used for the full-system test; contains phone numbers, so never commit it |

Column meanings are in [data_dictionary.md](data_dictionary.md).
