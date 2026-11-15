# Decision log

Each entry: context → decision → consequences. A decision can be revisited, but the change and the
reason get written here first.

## ADR-001 — Dataset: UCI Credit Card Default (Taiwan) — *Accepted*
Context: I wanted a clean public dataset for v1 and considered Give Me Some Credit and UCI Taiwan.
Give Me Some Credit is a single snapshot with no exposure amounts and no month-by-month history. UCI
Taiwan has six months of repayment status, bills and payments per client.
Decision: use UCI Taiwan. It supports the PD scorecard, an EAD proxy (bill vs limit) and an
**empirical** roll-rate analysis from the monthly `PAY_*` sequences, which is the collections layer
that makes this project different from the usual scorecard exercise.
Consequences: this dataset has been analysed many times, so the value here is in the method and the
collections layer, not in the headline AUC. See ADR-003/004 for what the data cannot support. A
larger, messier dataset (Home Credit, Lending Club) is a possible v2.

## ADR-002 — Exclude sensitive attributes from model features — *Accepted*
Decision: `SEX` and `MARRIAGE` are not model inputs. They are used only to audit outcomes (Phase 12).
`AGE` and `EDUCATION` are candidate features only with a written justification (decided in Phase 4
with their IV on the table) and are included in the fairness audit either way.
Consequence: slightly lower raw performance may result; report it, do not hide it.

## ADR-003 — Validation design without a time dimension — *Accepted*
Decision: stratified 80/20 hold-out (test touched once) + repeated stratified 5-fold CV (3 repeats)
on the training part for selection and tuning. Genuine out-of-time validation is impossible with
this data, and the README says so.
Stability/drift analysis (Phase 11) uses **declared synthetic scenarios**. A shifted-window
pseudo-vintage design (features from months 2–6 → delinquency in month 1) is a possible addition,
to be decided in Phase 11. Whichever is used must be labelled as a simulation of drift, not
observed drift.

## ADR-004 — EAD proxy and LGD as assumptions — *Accepted*
Decision: EAD proxy = max(BILL_AMT1, 0) + CCF × max(LIMIT_BAL − BILL_AMT1, 0); CCF and LGD are
scenario grids in `config/assumptions.yaml`, never fitted. Expected loss is reported as a function of
those assumptions (sensitivity tables/heatmaps), and the README says so in plain words.
Alternative considered: fitting LGD on a second dataset with recoveries (e.g. Lending Club). Rejected
for v1 because it would mix two different populations and products in one loss figure.

## ADR-005 — WOE binning with `optbinning`; logistic regression as champion — *Accepted*
Decision: monotonic, constraint-aware WOE binning via `optbinning`; unit tests cross-check IV against
an independent computation. Champion = logistic regression on WOE features (interpretable, scorecard
compatible). Challenger = LightGBM. The champion is chosen on documented criteria, not on AUC alone.

## ADR-006 — PySpark stage runs in local Spark mode — *Accepted*
Context: the original plan was to run the PySpark pipeline in Databricks. That adds an external
platform that nobody cloning the repo can reproduce, and the dataset (30,000 rows) does not need a
cluster anyway.
Decision: the PySpark feature pipeline is developed, tested and executed with a local Spark session
(`local[*]`), with a parity test against the pandas implementation. The run log (Spark version,
row counts, parity result) is saved in `docs/evidence/`.
Consequences: everything in the project runs from a clean clone on one machine. The README must say
plainly that Spark ran in local mode: this stage shows the pipeline is portable to Spark, it does
not show performance at scale. Running the same code on a managed cluster is left as optional
future work.

## ADR-007 — Raw data committed to the repository — *Accepted*
Context: results should be reproducible from a clean clone without a manual download step, and the
file is small (about 5.5 MB). The licence (CC BY 4.0) allows redistribution with attribution.
Decision: commit the raw file unchanged, with its SHA-256 recorded in `docs/DATA.md`.

## ADR-008 — Treatment of data anomalies — *Accepted*
Context: the data-quality checks (`notebooks/01_eda.ipynb`, numbers in
`reports/metrics/data_quality.json`) found codes that the dataset documentation does not describe,
plus negative balances, inactive accounts, identical rows and long-tailed amounts. Each needs a
decision before modelling. The train/test split did not exist yet when these decisions were taken,
so they were taken without looking at default rates by code or by feature. The target was used only
for the overall default count and to count identical rows with and without the target column.

Decisions:
- **`EDUCATION` 0, 5, 6 → 4 ("others"); `MARRIAGE` 0 → 3 ("others").** 345 rows (1.15%) and 54 rows
  (0.18%). The documentation gives no meaning for these codes, and "others" is the only documented
  category that does not claim one. The mapping is fixed in `creditrisk.data.clean`, does not depend
  on the data, and fails on any code it has not seen.
- **`PAY_*` codes −2 and 0 are kept as separate categories.** They are not merged with −1 and not
  placed on the months-of-delay scale. Their meaning is an **ASSUMPTION** (no documentation and no
  citation). The evidence comes from May–August, pairing the status of a month with that month's
  statement and the payment made the month after (see ADR-009 for why this pairing). Code 0: 2.0% of
  account-months have no positive statement and 0.2% of positive statements are paid in full, which
  fits "revolving credit". Code −2: 63.4% have no positive statement, and 93.5% of the positive ones
  are paid in full, so it reads as "no balance, or balance cleared", which is wider than the usual
  "no consumption".
- **The September status (`PAY_1`) is not on the same footing as the earlier months.** Code 1 (one
  month of delay) appears 3,688 times in `PAY_1` and 28, 4, 2, 0 and 0 times in `PAY_2`…`PAY_6`,
  while code 2 is common in every month. In September, 1,689 accounts whose bill is zero or negative
  carry a delay code; in the earlier months that happens in 22 to 125 accounts, and such accounts
  are coded −2 instead. A large part of the September "one month late" group is therefore not a
  delinquency that can be seen in the bills. Nothing is recoded. Phase 4 decides how features that
  compare months treat `PAY_1`, and Phase 10 must state this as a limitation of the roll-rate
  analysis: transitions into September cannot be read like the other months.
- **Negative bills and bills above the limit are kept.** Negative bills occur in 590 to 688 accounts
  per month and bills above `LIMIT_BAL` in 798 to 2,115. The file does not define them; they are
  read as credit balances and over-limit accounts (interpretation). A floor at zero is applied only
  inside formulas that need it (utilisation, the EAD proxy of ADR-004).
- **Inactive accounts stay in the population.** 795 accounts have every bill and every payment at
  zero in the six months (866 have every bill at zero). They hold an open credit line, so a bank
  would score them, and removing them would change the population every later result refers to.
  495 of the 795 carry a delay code in `PAY_1` with nothing ever billed (same anomaly as above).
- **Identical rows are kept.** The raw `ID` column is a row number (1 to 30,000) and is not
  documented as a client key; the loader drops it. Without it, 35 rows (0.12%) repeat an earlier row
  in every column. 30 of them are inactive accounts, which are bound to look alike. The other 5 have
  bills or payments and may be repeated records, but nothing in the file confirms it, and dropping
  rows on a guess would change the population. Consequence for Phase 3: 227 rows have a twin on the
  candidate model features, so twins can fall on both sides of the train/test split. Phase 3 counts
  how many do and saves the count.
- **No value is removed or capped.** Bills and payments have long right tails, and there is no way
  to tell an error from a large balance. Binned features and tree models are not sensitive to the
  size of an extreme value. The percentiles saved in `data_quality.json` are descriptive only: any
  threshold a later step needs is fitted on training data there.

Consequences: the only change to the raw values is the `EDUCATION`/`MARRIAGE` recoding. The meaning
of −2 and 0 stays an assumption, and Phase 10 must show how its results change under a different
mapping of these codes to delinquency buckets.

## ADR-009 — Payment timing and information available at prediction time — *Accepted*
Context: the file does not say which statement a payment settles, which statement and payment a
repayment status refers to, or on which date any column was recorded. Feature engineering needs the
first, and "no feature uses information from the target period" needs the other two.

Evidence (`reports/metrics/data_quality.json`):
- Among accounts that paid something in a month, the payment equals the previous month's statement
  in 16.4% to 19.8% of cases and the same month's statement in 2.5% to 2.7%; nearly all of the
  latter are accounts whose two statements are the same amount.
- On all accounts, the purchases implied by two consecutive statements and the payment between them
  are negative for about 0.4% of accounts per month if the payment settles the previous statement,
  and for 15.8% to 19.1% if it settles the same month's.
- For May–August, pairing the status of month k with statement k and the payment of the following
  month fits the documented codes much better than pairing it with the same month's payment: under
  code −1 ("paid duly") 98.4% of positive statements are paid in full, against 75.8%.

Decisions:
- `PAY_AMTk` is treated as the payment of statement `BILL_AMT{k+1}`. Payment ratios are built as
  `PAY_AMTk / BILL_AMT{k+1}`, which gives five usable months.
- The model is scored at the end of September 2005 to predict default in October 2005. All columns
  other than the target are candidate features, except `SEX` and `MARRIAGE` (ADR-002). That they
  were all known before October is an **ASSUMPTION** resting on the dataset description.
- **`PAY_1` is the column most at risk under that assumption.** If it followed the same rule as the
  earlier months it would be set by the October payment, which belongs to the outcome period and is
  not in the file. The anomalies in ADR-008 suggest it was instead recorded on a different basis,
  but the file cannot settle which. `PAY_1` stays a candidate feature. Phase 4 and Phase 5 must
  run, on training data only, a named leakage check: its information value against the 0.5 alarm
  threshold, and model performance with and without `PAY_1`, reported as a sensitivity.
- `LIMIT_BAL`, `AGE` and `EDUCATION` are not dated in the file either. A credit limit can be
  adjusted in response to behaviour, so `LIMIT_BAL` could reflect a later date than September. This
  cannot be checked and is stated as a limitation.

Consequences: if the `PAY_1` check raises the alarm, the treatment of `PAY_1` is decided again in
this log before any model is finalised. Alternative considered: excluding `PAY_1` now and modelling
on months 2–6 only. Rejected because it would discard the most recent behaviour without evidence
that it leaks.
