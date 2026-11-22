# Scorecard de riesgo crediticio y estrategia de cobranza

> Estado: **en construcción.** Este README se escribe en la fase final, y cada número que aparezca
> en él saldrá de `reports/metrics/*.json`. Alcance y plan: [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md).

## Qué es este proyecto
Un flujo completo de riesgo crediticio minorista sobre el dataset de UCI *Default of Credit Card
Clients*: scorecard de PD (WOE/IV + regresión logística) con un modelo retador de gradient boosting,
marco de pérdida esperada con supuestos explícitos, interpretabilidad, monitoreo de estabilidad,
auditoría de equidad y una capa de cobranza con roll rates y priorización, construida a partir del
historial mensual de mora observado.

## Qué no es
Un modelo de producción. El dataset no tiene fechas de originación, recuperaciones ni solicitantes
rechazados, así que la validación fuera de tiempo, una LGD estimada y la inferencia de rechazados
quedan fuera del alcance. Ver [`docs/DATA.md`](docs/DATA.md).

## Cómo reproducirlo
```bash
make install
make data-check FILE="<nombre del archivo crudo>"
make check
make notebooks
```

## Cita de los datos
Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive
accuracy of probability of default of credit card clients. *Expert Systems with Applications*,
36(2), 2473–2480.

## Autor
Caleb Samaniego Granados — Lima, Perú.
