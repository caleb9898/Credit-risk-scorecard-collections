# Project specification

## Objective
Build a bank-credible, end-to-end retail credit-risk workflow that a credit-risk professional could
audit and recognise as representative of real work, and that doubles as a base for fraud/pricing-style
projects later. Anchor: collections and credit risk.

## Scope (v1)
Included: PD scorecard (WOE/IV + logistic regression), challenger (LightGBM), scorecard scaling,
EL framework with explicit assumptions, SHAP + reason codes, stability monitoring, fairness audit,
collections roll-rate and prioritisation layer, business-impact analysis, Streamlit app, PySpark stage,
documentation.
Excluded from v1: reject inference, fitted LGD, true out-of-time validation, real-time serving,
MLOps beyond reproducibility. (Each is listed in the README as a limitation.)

## Dataset
UCI Default of Credit Card Clients (ADR-001). Details and quirks: `docs/DATA.md`.

## Phases
Each phase ends with: green `make check`, a review against `docs/RIGOR_STANDARDS.md`, and
`docs/DECISIONS.md` updated. Metrics are saved to `reports/metrics/`.

### P1 — Environment and data foundation
Deliver: raw file in `data/raw/`, hash recorded, `make data-check` passes, CI green, README stub.
Accept: a clean clone installs and passes `make check`; `docs/DATA.md` table filled.

### P2 — Data quality and EDA (`notebooks/01_eda.ipynb`)
Deliver: target rate, missingness, outliers, undocumented codes, negative balances, duplicates,
payment-timing check; `reports/metrics/data_quality.json`.
Accept: every anomaly has a recorded decision; features available at prediction time identified.

### P3 — Target, split, CV design
Deliver: stratified 80/20 hold-out by saved index, repeated stratified CV on train (ADR-003).
Accept: split reproducible; prevalence comparable; a test proves no index overlap; limitation stated.

### P4 — Feature engineering and WOE/IV (`02_woe_iv_binning.ipynb`)
Deliver: engineered features (e.g. utilisation, payment ratios, delinquency counts and severity, trends)
with a rationale each; monotonic WOE binning; IV table; correlation/VIF; WOE stability across folds.
Accept: binning fitted on train only; IV cross-checked by an independent implementation in tests;
features with IV > 0.5 investigated for leakage.

### P5 — PD model: logistic regression (`03_pd_model_logistic.ipynb`)
Deliver: statsmodels fit with SE/p-values/CIs; sign review; AUC/Gini/KS/PR-AUC with bootstrap CIs on
CV and on test (once); calibration summary and reliability plot; operating-point table.
Accept: signs explained; calibration reported honestly; metrics saved to JSON.

### P6 — Challenger: LightGBM (`04_model_comparison_gbm.ipynb`)
Deliver: CV-tuned LightGBM (documented search); paired bootstrap comparison with the champion;
discussion of performance vs interpretability and regulatory acceptability.
Accept: champion choice justified on stated criteria; no tuning on test.

### P7 — Scorecard (`05_scorecard_scaling.ipynb`)
Deliver: base score / base odds / PDO parameters; points table per bin; score bands with PD,
volume and rank-ordering check; `artifacts/final/scorecard_points.csv`.
Accept: a test shows points reproduce the logistic PD exactly (within numerical tolerance); bands monotonic.

### P8 — EAD, LGD and expected loss (`07_ead_lgd_expected_loss.ipynb`)
Deliver: EAD proxy and LGD scenarios (ADR-004); EL at account, band and portfolio level; sensitivity
heatmaps over CCF × LGD; plain-language statement of what is assumed.
Accept: every EL figure shows its assumptions; nothing labelled "estimated" that is assumed.

### P9 — Interpretability (`06_shap_interpretability.ipynb`)
Deliver: SHAP (global/local) for the challenger; scorecard point contributions and reason codes for the
champion; agreement analysis with IV ranking.
Accept: consistency check by code between additive contributions and model output; 2–3 worked examples.

### P10 — Collections roll-rate and prioritisation (`08_rollrate_collections_strategy.ipynb`)
Deliver: delinquency-bucket mapping from `PAY_*` codes (assumptions stated); monthly transition
matrices PAY_{k+1} → PAY_k with bootstrap CIs; cure/roll rates; predictive power of current bucket vs
PD score for next-month default; **capture-rate curves** at limited contact capacity (empirical);
treatment-segmentation matrix (score band × bucket); simulated treatment impact with sensitivity over
cure-uplift and cost assumptions (labelled SIMULATION / ASSUMPTION).
Accept: empirical and simulated parts are visibly separate in notebook, figures and README.

### P11 — Stability and stress (`09_psi_monitoring.ipynb`, `10_stress_test.ipynb`)
Deliver: PSI/CSI train vs test (sanity); declared drift scenarios and their effect on PSI and
performance (ADR-003); PD-shift stress scenarios and EL impact; monitoring policy with triggers.
Accept: all drift is labelled as simulated; thresholds and actions documented.

### P12 — Fairness and governance
Deliver: audit by SEX, AGE band, EDUCATION; draft `reports/model_card.md` (intended use, data, metrics,
limits, monitoring, owner).
Accept: findings reported with caveats; no compliance claims.

### P13 — Business impact (`11_business_impact.ipynb`)
Deliver: cut-off policy from cost-ratio grid; recall-oriented operating point; EL and default-rate
effect of the policy; 1–2 figures readable by a non-technical audience.
Accept: cost assumptions explicit; trade-off curves shown, not a single cherry-picked point.

### P14 — Streamlit app
Deliver: app reading `artifacts/final/`, inputs for a hypothetical client → PD, score, band, reason
codes; disclaimer that it is a methodology demo; app test.
Accept: runs from a clean clone; no hidden state; deployable on Streamlit Community Cloud.

### P15 — PySpark stage (local mode)
Deliver: PySpark implementation of the feature pipeline in `src/creditrisk/spark/` with a parity test
against the pandas implementation, run with a local Spark session (ADR-006); notes on what would
change at scale; run log saved in `docs/evidence/`.
Accept: parity test passes; evidence file present; README states that Spark ran in local mode.

### P16 — Documentation and release
Deliver: README (problem, data, method, results from JSON, differentiator, limitations, how to
reproduce), `docs/methodology.md`, final model card, `make notebooks` green, release tag `v1.0`.
Accept: every claim in the README traces to a saved metric; a stranger can reproduce results from
the README.

## Later (not v1)
Fraud satellite (IEEE-CIS or ULB) and pricing satellite (freMTPL), reusing `creditrisk.validation`.
