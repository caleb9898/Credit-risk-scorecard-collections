# Credit Risk Scorecard & Collections Strategy

> Status: **under construction.** This README is written in the final phase, and every number in it
> will come from `reports/metrics/*.json`. Scope and plan: [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md).

## What this project is
An end-to-end retail credit-risk workflow on the UCI *Default of Credit Card Clients* dataset:
PD scorecard (WOE/IV + logistic regression) with a gradient-boosting challenger, expected-loss
framework with explicit assumptions, interpretability, stability monitoring, fairness audit, and a
collections roll-rate and prioritisation layer built from the observed monthly delinquency history.

## What it is not
A production model. The dataset has no origination dates, no recoveries and no rejected applicants,
so out-of-time validation, fitted LGD and reject inference are out of scope. See
[`docs/DATA.md`](docs/DATA.md).

## Reproduce
```bash
make install
make data-check FILE="<raw file name>"
make check
make notebooks
```

## Data citation
Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive
accuracy of probability of default of credit card clients. *Expert Systems with Applications*,
36(2), 2473–2480.

## Author
Caleb Samaniego Granados — Lima, Peru.
