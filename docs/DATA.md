# Datos

## Fuente
UCI Machine Learning Repository — *Default of Credit Card Clients* (30,000 clientes de tarjeta de
crédito de un banco en Taiwán; comportamiento de abril a septiembre de 2005; target observado en
octubre de 2005).

Cita: Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the
predictive accuracy of probability of default of credit card clients. *Expert Systems with
Applications*, 36(2), 2473–2480.

Licencia: Creative Commons Attribution 4.0 International (CC BY 4.0), según la página del dataset
en UCI (revisada en la fecha de descarga indicada abajo).

## Cómo obtenerlos
1. Descargar el dataset desde su página en UCI (Dataset ID 350) y descomprimirlo.
2. Copiar el archivo sin cambios a `data/raw/` y subirlo al repositorio (pocos MB).
3. Ejecutar `make data-check FILE=<nombre del archivo>`; copiar el SHA-256 a `config/project.yaml`
   (`data.raw_sha256`) y a la tabla de abajo.

| Campo | Valor |
|---|---|
| Nombre del archivo | `default_of_credit_card_clients.xls` (renombrado desde `default of credit card clients.xls`, contenido sin cambios) |
| SHA-256 | `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933` |
| Fecha de descarga | 2026-11-15 |
| Licencia (según la página de UCI) | CC BY 4.0 |

## Esquema (nombres canónicos usados en este proyecto)
| Columna | Significado |
|---|---|
| LIMIT_BAL | Límite de crédito (NT$), crédito individual más el suplementario familiar |
| SEX, EDUCATION, MARRIAGE, AGE | Datos demográficos (SEX y MARRIAGE excluidos de los modelos, ADR-002) |
| PAY_1 … PAY_6 | Estado de pago, mes 1 = sep 2005 … mes 6 = abr 2005 |
| BILL_AMT1 … BILL_AMT6 | Monto del estado de cuenta (NT$), misma indexación de meses |
| PAY_AMT1 … PAY_AMT6 | Monto pagado en el mes (NT$), misma indexación de meses |
| DEFAULT | 1 = cayó en default en octubre de 2005 (target) |

Los archivos crudos usan los nombres `X1..X23, Y` o `PAY_0, PAY_2, …`; `creditrisk.data.schema`
lleva ambos a los nombres canónicos (el `PAY_0` de la fuente pasa a ser `PAY_1`).

## Target, ventana y población
30,000 clientes de tarjeta de crédito de un banco en Taiwán. Las variables describen abril a
septiembre de 2005; el target es el default en octubre de 2005. El archivo tiene 6,636 defaults
(22.12%), contados en el archivo y fijados en el cargador como control de regresión. La fuente no
dice cómo se definió "default" ni cómo se muestrearon los clientes, así que esta tasa no se lee como
la tasa de default de la cartera del banco.

## Particularidades de los datos y cómo se tratan
Encontradas en `notebooks/01_eda.ipynb`; los conteos están guardados en
`reports/metrics/data_quality.json` y las decisiones en `docs/DECISIONS.md` (ADR-008, ADR-009).

| Particularidad | Lo que muestra el archivo | Tratamiento |
|---|---|---|
| Columna `ID` | Número de fila de 1 a 30,000, no documentado como clave de cliente | El cargador la descarta |
| Valores nulos | Ninguno | Nada que hacer |
| `EDUCATION` códigos 0, 5, 6 | 345 filas fuera del rango documentado 1–4 | Recodificadas a 4 ("otros") |
| `MARRIAGE` código 0 | 54 filas fuera del rango documentado 1–3 | Recodificadas a 3 ("otros") |
| `PAY_*` códigos −2 y 0 | No documentados, aunque 0 es el valor más frecuente en todos los meses | Se conservan como categorías propias; su significado es un SUPUESTO (−2 = sin saldo o saldo cancelado, 0 = crédito revolvente) |
| `PAY_*` código 1 | 3,688 cuentas en `PAY_1`, como máximo 28 en cualquier mes anterior | Se conserva; `PAY_1` no es directamente comparable con `PAY_2`…`PAY_6` |
| Códigos de atraso sin nada facturado | 1,689 cuentas en septiembre, de 22 a 125 en los meses anteriores | Se conservan; limitación para el análisis de roll rates |
| `PAY_*` código 9 | Documentado pero nunca presente | Nada que hacer |
| Momento de registro de `PAY_1` | No se puede demostrar que sea anterior al resultado de octubre | Variable candidata; su disponibilidad es un SUPUESTO con una verificación de fuga sobre datos de entrenamiento |
| `BILL_AMT*` negativo | De 590 a 688 cuentas por mes (leído como saldo a favor) | Se conserva; se trunca a cero solo dentro de las fórmulas que lo necesitan |
| `BILL_AMT*` mayor que `LIMIT_BAL` | De 798 a 2,115 cuentas por mes | Se conserva |
| Cuentas inactivas | 795 cuentas con todas las facturas y pagos en cero | Se conservan en la población |
| Filas idénticas | 35 filas repiten una fila anterior una vez descartado `ID`; 30 de ellas son cuentas inactivas | Se conservan |
| Montos extremos | Colas derechas largas en facturas y pagos | Se conservan, sin truncar |
| Timing de pagos | Los pagos completos coinciden con el estado de cuenta del mes anterior mucho más a menudo que con el del mismo mes | `PAY_AMTk` cancela `BILL_AMT{k+1}` |

La recodificación está en `creditrisk.data.clean`. Es el único cambio que se hace a los valores
crudos.

## Lo que este dataset no permite (declararlo en el README)
- Sin fecha de originación → **no hay partición fuera de tiempo genuina** (ADR-003).
- Sin saldo al momento del default ni recuperaciones → **EAD y LGD no se pueden estimar**, solo
  suponer (ADR-004).
- Sin solicitantes rechazados → **no hay inferencia de rechazados**.
- Un solo banco, un solo mercado, 2005 → los resultados no se generalizan a las carteras actuales;
  esto es una demostración metodológica, no un modelo de producción.
