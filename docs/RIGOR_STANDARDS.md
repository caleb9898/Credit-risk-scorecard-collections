# Rigor standards (the validator's checklist)

A phase is not done until every applicable item is satisfied or an exception is written down in
`docs/DECISIONS.md`. Reference ranges are **sanity checks, never targets** — do not tune toward them.

## 1. Data
- [ ] Raw file hash recorded; schema validated by code, not by eye.
- [ ] Every anomaly (undocumented codes, negative balances, duplicates) has a recorded decision.
- [ ] Target definition, observation window and population are stated in one place.

## 2. Splitting and leakage
- [ ] Hold-out split created once, saved by index, seeded; prevalence of the event matches across splits.
- [ ] All fitted transformations are fitted on training data only (inside CV folds when tuning).
- [ ] No feature uses information from the target period. Features are derived only from the candidate
      columns of ADR-009 (limit, age, education and the PAY/BILL/PAY_AMT months that precede the October
      outcome); the leakage check on `PAY_1` required by ADR-009 is run on training data.
- [ ] Test set evaluated once per final model; nothing changed afterwards because of it.
- [ ] Suspicious performance is investigated: an AUC far above what public analyses of this dataset
      report (roughly the high-0.70s) is a leakage alarm, not a success.

## 3. Binning, WOE and IV
- [ ] Binning monotonic where business logic demands it; minimum bin size stated; missing/special codes
      handled as explicit bins.
- [ ] IV table reported with the usual bands (<0.02 useless, 0.02–0.1 weak, 0.1–0.3 medium,
      0.3–0.5 strong, >0.5 suspicious → check leakage).
- [ ] WOE/bin stability checked across CV folds.
- [ ] Multicollinearity assessed (pairwise correlation and VIF) with a stated threshold.

## 4. Model
- [ ] Coefficient signs agree with business logic; any exception is explained.
- [ ] Standard errors / p-values / confidence intervals for the logistic model (statsmodels).
- [ ] Hyperparameters selected by CV on training data only; search space and budget documented.
- [ ] Champion vs challenger compared with uncertainty (bootstrap CIs for AUC/KS and for the paired
      difference), not on point estimates alone.

## 5. Validation
- [ ] Discrimination: AUC, Gini, KS **with bootstrap CIs**; PR-AUC given the 22% event rate.
- [ ] Calibration: Brier, calibration intercept/slope, reliability plot by decile.
- [ ] Operating points: recall, precision, flag rate at stated thresholds. In collections, recall (not
      missing delinquent accounts) is the prioritised metric; justify the threshold from costs.
- [ ] Rank-ordering: event rate by score band is monotonic; bands have adequate size.
- [ ] Stability: PSI (score) and CSI (characteristics) with thresholds (<0.10 stable, 0.10–0.25 moderate,
      >0.25 major) and a redevelopment-trigger policy.

## 6. Interpretability and fairness
- [ ] Global and local explanations; reason codes for the top adverse factors per account.
- [ ] Explanations agree with the scorecard's own additive structure (consistency check by code).
- [ ] Fairness audit on SEX, AGE bands, EDUCATION: AUC, calibration, flag rate and recall by group.
      Report differences as findings with caveats; do not claim legal compliance.

## 7. Expected loss and collections
- [ ] EL = PD × EAD × LGD with EAD/LGD labelled ASSUMPTION; sensitivity grid reported.
- [ ] Roll-rate matrices computed from observed monthly `PAY_*` transitions, with uncertainty
      (bootstrap) and a documented mapping of codes to delinquency buckets.
- [ ] Prioritisation results (capture rate of future defaulters at a given contact capacity) are
      **empirical**; anything involving treatment effects or costs is **simulated** and labelled so.

## 8. Reproducibility and reporting
- [ ] `make check` and `make notebooks` pass on a clean clone; seeds fixed; outputs deterministic.
- [ ] Every number in prose is traceable to `reports/metrics/*.json`.
- [ ] Limitations section is specific (see `docs/DATA.md`, "What this dataset cannot support").
- [ ] No unsupported superlatives ("state-of-the-art", "excellent"). Describe, don't sell.
