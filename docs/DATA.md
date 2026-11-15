# Data

## Source
UCI Machine Learning Repository — *Default of Credit Card Clients* (30,000 credit-card clients of a
bank in Taiwan; behaviour April–September 2005; target observed in October 2005).

Citation: Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the
predictive accuracy of probability of default of credit card clients. *Expert Systems with
Applications*, 36(2), 2473–2480.

Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the dataset's UCI
page (checked on the download date below).

## How to obtain
1. Download the dataset from its UCI page (Dataset ID 350) and unzip it.
2. Copy the file unchanged into `data/raw/` and commit it (≈ few MB).
3. Run `make data-check FILE=<file name>`; copy the SHA-256 into `config/project.yaml`
   (`data.raw_sha256`) and below.

| Field | Value |
|---|---|
| File name | `default_of_credit_card_clients.xls` (renamed from `default of credit card clients.xls`, content unchanged) |
| SHA-256 | `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933` |
| Download date | 2026-11-15 |
| Licence (as shown on UCI page) | CC BY 4.0 |

## Schema (canonical names used in this project)
| Column | Meaning |
|---|---|
| LIMIT_BAL | Credit limit (NT$), individual + family supplementary credit |
| SEX, EDUCATION, MARRIAGE, AGE | Demographics (SEX/MARRIAGE excluded from models, ADR-002) |
| PAY_1 … PAY_6 | Repayment status, month 1 = Sep 2005 … month 6 = Apr 2005 |
| BILL_AMT1 … BILL_AMT6 | Bill statement amount (NT$), same month indexing |
| PAY_AMT1 … PAY_AMT6 | Amount paid in the month (NT$), same month indexing |
| DEFAULT | 1 = defaulted in October 2005 (target) |

Raw files use either `X1..X23, Y` or `PAY_0, PAY_2, …` names; `creditrisk.data.schema` maps both to
the canonical names (the source's `PAY_0` becomes `PAY_1`).

## Known quirks to resolve in Phase 2 (each needs a recorded decision)
- `EDUCATION` contains values outside the documented 1–4 (observed in public analyses: 0, 5, 6).
- `MARRIAGE` contains 0, outside the documented 1–3.
- `PAY_*` documents −1 (paid duly) and 1–9 (months of delay), but −2 and 0 also appear. Common
  interpretations: −2 = no consumption, 0 = revolving credit / minimum payment. These are
  **interpretations, not documentation**; state them as assumptions and test sensitivity.
- `BILL_AMT*` can be negative (credit balance). Decide treatment and document it.
- Timing of `PAY_AMTk` vs `BILL_AMT`: a payment in month k presumably settles the *previous*
  statement (`BILL_AMT{k+1}`). Verify empirically before building payment-ratio features.
- Duplicates / identical rows, and the exact default count (expected 6,636 of 30,000 — if the file
  disagrees, investigate; never adjust the expectation silently).

## What this dataset cannot support (state these in the README)
- No origination date → **no genuine out-of-time split** (ADR-003).
- No balance at default and no recoveries → **EAD and LGD cannot be estimated**, only assumed (ADR-004).
- No rejected applicants → **no reject inference**.
- Single bank, single market, 2005 → results do not generalise to today's portfolios; this is a
  methodology demonstration, not a production model.
