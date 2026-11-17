# Estándares de rigor (la lista de control del validador)

Una fase no está terminada hasta que cada punto aplicable se cumple o se registra una excepción en
`docs/DECISIONS.md`. Los rangos de referencia son **controles de cordura, nunca objetivos**: no se
ajusta el modelo para alcanzarlos.

## 1. Datos
- [ ] Hash del archivo crudo registrado; esquema validado por código, no a ojo.
- [ ] Cada anomalía (códigos no documentados, saldos negativos, duplicados) tiene una decisión
      registrada.
- [ ] La definición del target, la ventana de observación y la población están declaradas en un
      solo lugar.

## 2. Partición y fuga de información
- [ ] Hold-out creado una sola vez, guardado por índice y con semilla; la prevalencia del evento
      coincide entre las particiones.
- [ ] Toda transformación ajustada se ajusta solo con datos de entrenamiento (dentro de los folds de
      validación cruzada cuando se afinan hiperparámetros).
- [ ] Ninguna variable usa información del período del target. Las variables se derivan solo de las
      columnas candidatas de ADR-009 (límite, edad, educación y los meses de PAY/BILL/PAY_AMT
      anteriores al resultado de octubre); la verificación de fuga sobre `PAY_1` que exige ADR-009
      se ejecuta con datos de entrenamiento.
- [ ] El conjunto de test se evalúa una vez por modelo final; nada se cambia después a causa de él.
- [ ] Un desempeño sospechoso se investiga: un AUC muy por encima de lo que reportan los análisis
      públicos de este dataset (aproximadamente la parte alta de los 0.70) es una alarma de fuga, no
      un éxito.

## 3. Binning, WOE e IV
- [ ] Binning monótono donde la lógica de negocio lo exige; tamaño mínimo de bin declarado; códigos
      faltantes o especiales tratados como bins explícitos.
- [ ] Tabla de IV reportada con las bandas habituales (<0.02 inútil, 0.02–0.1 débil, 0.1–0.3 medio,
      0.3–0.5 fuerte, >0.5 sospechoso → revisar fuga).
- [ ] Estabilidad del WOE y de los bins verificada entre folds de validación cruzada.
- [ ] Multicolinealidad evaluada (correlación por pares y VIF) con un umbral declarado.

## 4. Modelo
- [ ] Los signos de los coeficientes concuerdan con la lógica de negocio; toda excepción se explica.
- [ ] Errores estándar, p-valores e intervalos de confianza para el modelo logístico (statsmodels).
- [ ] Hiperparámetros elegidos por validación cruzada solo con datos de entrenamiento; espacio de
      búsqueda y presupuesto documentados.
- [ ] Campeón y retador comparados con incertidumbre (intervalos bootstrap para AUC/KS y para la
      diferencia pareada), no solo con estimaciones puntuales.

## 5. Validación
- [ ] Discriminación: AUC, Gini, KS **con intervalos bootstrap**; PR-AUC dada la tasa de evento del 22%.
- [ ] Calibración: Brier, intercepto y pendiente de calibración, gráfico de confiabilidad por decil.
- [ ] Puntos de operación: recall, precisión y tasa de marcado en umbrales declarados. En cobranza,
      el recall (no dejar pasar cuentas que caerán en mora) es la métrica prioritaria; el umbral se
      justifica con costos.
- [ ] Ordenamiento: la tasa de evento por banda de score es monótona; las bandas tienen tamaño
      suficiente.
- [ ] Estabilidad: PSI (score) y CSI (características) con umbrales (<0.10 estable, 0.10–0.25
      moderado, >0.25 mayor) y una política de disparadores para redesarrollo.

## 6. Interpretabilidad y equidad
- [ ] Explicaciones globales y locales; códigos de razón para los principales factores adversos por
      cuenta.
- [ ] Las explicaciones concuerdan con la estructura aditiva del propio scorecard (verificación de
      consistencia por código).
- [ ] Auditoría de equidad sobre SEX, tramos de AGE y EDUCATION: AUC, calibración, tasa de marcado
      y recall por grupo. Las diferencias se reportan como hallazgos con sus salvedades; no se afirma
      cumplimiento legal.

## 7. Pérdida esperada y cobranza
- [ ] EL = PD × EAD × LGD con EAD/LGD etiquetadas SUPUESTO; grilla de sensibilidad reportada.
- [ ] Matrices de roll rate calculadas a partir de las transiciones mensuales observadas de `PAY_*`,
      con incertidumbre (bootstrap) y un mapeo documentado de códigos a tramos de mora.
- [ ] Los resultados de priorización (tasa de captura de futuros morosos a una capacidad de contacto
      dada) son **empíricos**; todo lo que involucre efectos de tratamiento o costos es **simulado**
      y se etiqueta así.

## 8. Reproducibilidad y reporte
- [ ] `make check` y `make notebooks` pasan en un clon limpio; semillas fijas; salidas deterministas.
- [ ] Cada número en el texto se rastrea hasta `reports/metrics/*.json`.
- [ ] La sección de limitaciones es específica (ver `docs/DATA.md`, "Lo que este dataset no permite").
- [ ] Sin superlativos sin sustento ("estado del arte", "excelente"). Describir, no vender.
