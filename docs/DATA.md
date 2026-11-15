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

## Target, window and population
30,000 credit-card clients of one bank in Taiwan. Features describe April–September 2005; the target
is default in October 2005. The file has 6,636 defaults (22.12%), counted from the file
and pinned in the loader as a regression check. The source does not say how "default" was defined
or how the clients were sampled, so this rate is not read as the default rate of the bank's
portfolio.

## Data quirks and how they are handled
Found by `notebooks/01_eda.ipynb`; counts are saved in `reports/metrics/data_quality.json` and the
decisions are in `docs/DECISIONS.md` (ADR-008, ADR-009).

| Quirk | What the file shows | Handling |
|---|---|---|
| `ID` column | Row number 1 to 30,000, not documented as a client key | Dropped by the loader |
| Null values | None | Nothing to do |
| `EDUCATION` codes 0, 5, 6 | 345 rows outside the documented 1–4 | Recoded to 4 ("others") |
| `MARRIAGE` code 0 | 54 rows outside the documented 1–3 | Recoded to 3 ("others") |
| `PAY_*` codes −2 and 0 | Not documented, yet 0 is the most common value in every month | Kept as separate categories; meaning is an ASSUMPTION (−2 = no balance or balance cleared, 0 = revolving credit) |
| `PAY_*` code 1 | 3,688 accounts in `PAY_1`, at most 28 in any earlier month | Kept; `PAY_1` is not directly comparable with `PAY_2`…`PAY_6` |
| Delay codes with nothing billed | 1,689 accounts in September, 22 to 125 in earlier months | Kept; limitation for the roll-rate analysis |
| `PAY_*` code 9 | Documented but never present | Nothing to do |
| Timing of `PAY_1` | Cannot be shown to precede the October outcome | Candidate feature; availability is an ASSUMPTION with a leakage check on training data |
| Negative `BILL_AMT*` | 590 to 688 accounts per month (read as credit balances) | Kept; floored at zero only inside formulas that need it |
| `BILL_AMT*` above `LIMIT_BAL` | 798 to 2,115 accounts per month | Kept |
| Inactive accounts | 795 accounts with every bill and payment at zero | Kept in the population |
| Identical rows | 35 rows repeat an earlier row once `ID` is dropped; 30 of them are inactive accounts | Kept |
| Extreme amounts | Long right tails in bills and payments | Kept, no capping |
| Payment timing | Full payments match the previous month's statement far more often than the same month's | `PAY_AMTk` settles `BILL_AMT{k+1}` |

The recoding lives in `creditrisk.data.clean`. It is the only change made to the raw values.

## What this dataset cannot support (state these in the README)
- No origination date → **no genuine out-of-time split** (ADR-003).
- No balance at default and no recoveries → **EAD and LGD cannot be estimated**, only assumed (ADR-004).
- No rejected applicants → **no reject inference**.
- Single bank, single market, 2005 → results do not generalise to today's portfolios; this is a
  methodology demonstration, not a production model.
