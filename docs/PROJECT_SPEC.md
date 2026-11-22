# Especificación del proyecto

## Objetivo
Construir un flujo completo de riesgo crediticio minorista, creíble para un banco, que un profesional
de riesgo pueda auditar y reconocer como representativo del trabajo real, y que sirva después como
base para proyectos de fraude o pricing. Eje: cobranza y riesgo crediticio.

## Alcance (v1)
Incluido: scorecard de PD (WOE/IV + regresión logística), modelo retador (LightGBM), escalamiento
del scorecard, marco de pérdida esperada (EL) con supuestos explícitos, SHAP y códigos de razón,
monitoreo de estabilidad, auditoría de equidad, capa de cobranza con roll rates y priorización,
análisis de impacto de negocio, app en Streamlit, etapa en PySpark y documentación.
Excluido de la v1: inferencia de rechazados, LGD estimada, validación fuera de tiempo genuina,
servicio en tiempo real y MLOps más allá de la reproducibilidad. (Cada punto aparece en el README
como limitación.)

## Dataset
UCI Default of Credit Card Clients (ADR-001). Detalles y particularidades: `docs/DATA.md`.

## Fases
Cada fase termina con: `make check` en verde, una revisión contra `docs/RIGOR_STANDARDS.md` y
`docs/DECISIONS.md` actualizado. Las métricas se guardan en `reports/metrics/`.

### P1 — Entorno y base de datos
Entregables: archivo crudo en `data/raw/`, hash registrado, `make data-check` pasa, CI en verde,
README inicial.
Aceptación: un clon limpio se instala y pasa `make check`; tabla de `docs/DATA.md` completa.

### P2 — Calidad de datos y EDA (`notebooks/01_eda.ipynb`)
Entregables: tasa del target, valores faltantes, valores extremos, códigos no documentados, saldos
negativos, duplicados, verificación del timing de pagos; `reports/metrics/data_quality.json`.
Aceptación: cada anomalía tiene una decisión registrada; variables disponibles al momento de
predecir identificadas.

### P3 — Target, partición y diseño de validación cruzada
Entregables: hold-out estratificado 80/20 guardado por índice, validación cruzada estratificada
repetida sobre train (ADR-003).
Aceptación: partición reproducible; prevalencia comparable; un test prueba que no hay índices
repetidos entre train y test; limitación declarada.

### P4 — Ingeniería de variables y WOE/IV (`02_woe_iv_binning.ipynb`)
Entregables: variables construidas (por ejemplo utilización, ratios de pago, conteos y severidad de
mora, tendencias), cada una con su justificación; binning WOE monótono; tabla de IV;
correlación/VIF; estabilidad del WOE entre folds.
Aceptación: binning ajustado solo sobre train; IV verificado en los tests con una implementación
independiente; variables con IV > 0.5 investigadas por posible fuga de información.

### P5 — Modelo de PD: regresión logística (`03_pd_model_logistic.ipynb`)
Entregables: ajuste con statsmodels con errores estándar, p-valores e intervalos de confianza;
revisión de signos; AUC/Gini/KS/PR-AUC con intervalos bootstrap en validación cruzada y en test
(una sola vez); resumen de calibración y gráfico de confiabilidad; tabla de puntos de operación.
Aceptación: signos explicados; calibración reportada con honestidad; métricas guardadas en JSON.

### P6 — Retador: LightGBM (`04_model_comparison_gbm.ipynb`)
Entregables: LightGBM ajustado por validación cruzada (búsqueda documentada); comparación bootstrap
pareada con el campeón; discusión de desempeño frente a interpretabilidad y aceptabilidad
regulatoria.
Aceptación: elección del campeón justificada con criterios declarados; ningún ajuste sobre test.

### P7 — Scorecard (`05_scorecard_scaling.ipynb`)
Entregables: parámetros de puntaje base, odds base y PDO; tabla de puntos por bin; bandas de puntaje
con PD, volumen y verificación de ordenamiento; `artifacts/final/scorecard_points.csv`.
Aceptación: un test muestra que los puntos reproducen exactamente la PD de la logística (dentro de
la tolerancia numérica); bandas monótonas.

### P8 — EAD, LGD y pérdida esperada (`07_ead_lgd_expected_loss.ipynb`)
Entregables: proxy de EAD y escenarios de LGD (ADR-004); EL a nivel de cuenta, banda y cartera;
mapas de sensibilidad sobre CCF × LGD; declaración en lenguaje llano de lo que se supone.
Aceptación: cada cifra de EL muestra sus supuestos; nada supuesto se etiqueta como "estimado".

### P9 — Interpretabilidad (`06_shap_interpretability.ipynb`)
Entregables: SHAP (global y local) para el retador; contribuciones en puntos del scorecard y códigos
de razón para el campeón; análisis de concordancia con el ranking de IV.
Aceptación: verificación por código de la consistencia entre las contribuciones aditivas y la
salida del modelo; 2–3 ejemplos desarrollados.

### P10 — Roll rates de cobranza y priorización (`08_rollrate_collections_strategy.ipynb`)
Entregables: mapeo de los códigos `PAY_*` a tramos de mora (supuestos declarados); matrices de
transición mensuales PAY_{k+1} → PAY_k con intervalos bootstrap; tasas de cura y de deterioro; poder
predictivo del tramo actual frente al score de PD para el default del mes siguiente; **curvas de
captura** con capacidad de contacto limitada (empíricas); matriz de segmentación de tratamiento
(banda de score × tramo); impacto simulado del tratamiento con sensibilidad sobre los supuestos de
mejora en la cura y de costos (etiquetado SIMULACIÓN / SUPUESTO).
Aceptación: las partes empíricas y simuladas están visiblemente separadas en el notebook, las
figuras y el README.

### P11 — Estabilidad y estrés (`09_psi_monitoring.ipynb`, `10_stress_test.ipynb`)
Entregables: PSI/CSI de train frente a test (control de cordura); escenarios de deriva declarados y
su efecto en el PSI y el desempeño (ADR-003); escenarios de estrés con desplazamiento de PD e impacto
en EL; política de monitoreo con disparadores.
Aceptación: toda deriva está etiquetada como simulada; umbrales y acciones documentados.

### P12 — Equidad y gobierno
Entregables: auditoría por SEX, tramo de AGE y EDUCATION; borrador de `reports/model_card.md` (uso
previsto, datos, métricas, límites, monitoreo, responsable).
Aceptación: hallazgos reportados con sus salvedades; ninguna afirmación de cumplimiento normativo.

### P13 — Impacto de negocio (`11_business_impact.ipynb`)
Entregables: política de punto de corte a partir de una grilla de ratios de costo; punto de
operación orientado a recall; efecto de la política en EL y en la tasa de default; 1–2 figuras
legibles para una audiencia no técnica.
Aceptación: supuestos de costo explícitos; se muestran curvas de compensación, no un único punto
elegido a conveniencia.

### P14 — App en Streamlit
Entregables: app que lee `artifacts/final/`, con entradas para un cliente hipotético → PD, puntaje,
banda y códigos de razón; aviso de que es una demostración metodológica; test de la app.
Aceptación: corre desde un clon limpio; sin estado oculto; desplegable en Streamlit Community Cloud.

### P15 — Etapa en PySpark (modo local)
Entregables: implementación en PySpark del pipeline de variables en `src/creditrisk/spark/` con un
test de paridad contra la implementación en pandas, ejecutada con una sesión local de Spark
(ADR-006); notas sobre lo que cambiaría a escala; registro de ejecución guardado en `docs/evidence/`.
Aceptación: el test de paridad pasa; el archivo de evidencia existe; el README indica que Spark
corrió en modo local.

### P16 — Documentación y publicación
Entregables: README (problema, datos, método, resultados desde los JSON, diferenciador, limitaciones,
cómo reproducir), `docs/methodology.md`, model card final, `make notebooks` en verde, etiqueta de
versión `v1.0`.
Aceptación: cada afirmación del README se rastrea hasta una métrica guardada; un desconocido puede
reproducir los resultados a partir del README.

## Más adelante (fuera de la v1)
Satélite de fraude (IEEE-CIS o ULB) y satélite de pricing (freMTPL), reutilizando
`creditrisk.validation`.
