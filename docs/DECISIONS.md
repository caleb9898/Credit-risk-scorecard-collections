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
