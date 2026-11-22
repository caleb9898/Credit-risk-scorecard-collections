# Registro de decisiones

Cada entrada: contexto → decisión → consecuencias. Una decisión se puede revisar, pero el cambio y
su motivo se escriben aquí primero.

## ADR-001 — Dataset: UCI Credit Card Default (Taiwán) — *Aceptada*
Contexto: quería un dataset público y limpio para la v1 y consideré Give Me Some Credit y UCI Taiwán.
Give Me Some Credit es una sola foto, sin montos de exposición ni historial mes a mes. UCI Taiwán
tiene seis meses de estado de pago, facturas y pagos por cliente.
Decisión: usar UCI Taiwán. Permite el scorecard de PD, un proxy de EAD (factura frente a límite) y
un análisis **empírico** de roll rates a partir de las secuencias mensuales de `PAY_*`, que es la
capa de cobranza que distingue este proyecto del ejercicio habitual de scorecard.
Consecuencias: este dataset se ha analizado muchas veces, así que el valor está en el método y en la
capa de cobranza, no en el AUC principal. Ver ADR-003/004 para lo que los datos no permiten. Un
dataset más grande y desordenado (Home Credit, Lending Club) es una posible v2.

## ADR-002 — Excluir los atributos sensibles de las variables del modelo — *Aceptada*
Decisión: `SEX` y `MARRIAGE` no son entradas del modelo. Se usan solo para auditar resultados
(Fase 12). `AGE` y `EDUCATION` son variables candidatas solo con una justificación escrita (se decide
en la Fase 4 con su IV sobre la mesa) y se incluyen en la auditoría de equidad en cualquier caso.
Consecuencia: el desempeño bruto puede ser algo menor; se reporta, no se oculta.

## ADR-003 — Diseño de validación sin dimensión temporal — *Aceptada*
Decisión: hold-out estratificado 80/20 (el test se toca una sola vez) + validación cruzada
estratificada de 5 folds repetida (3 repeticiones) sobre la parte de entrenamiento, para selección y
ajuste. Una validación fuera de tiempo genuina es imposible con estos datos, y el README lo dice.
El análisis de estabilidad y deriva (Fase 11) usa **escenarios sintéticos declarados**. Un diseño de
pseudo-cosecha con ventana desplazada (variables de los meses 2–6 → mora en el mes 1) es un posible
añadido, a decidir en la Fase 11. El que se use debe etiquetarse como simulación de deriva, no como
deriva observada.

## ADR-004 — Proxy de EAD y LGD como supuestos — *Aceptada*
Decisión: proxy de EAD = max(BILL_AMT1, 0) + CCF × max(LIMIT_BAL − BILL_AMT1, 0); CCF y LGD son
grillas de escenarios en `config/assumptions.yaml`, nunca se estiman. La pérdida esperada se reporta
como función de esos supuestos (tablas y mapas de sensibilidad), y el README lo dice en palabras
simples.
Alternativa considerada: estimar la LGD con un segundo dataset que tenga recuperaciones (por ejemplo
Lending Club). Rechazada para la v1 porque mezclaría dos poblaciones y productos distintos en una
misma cifra de pérdida.

## ADR-005 — Binning WOE con `optbinning`; regresión logística como campeón — *Aceptada*
Decisión: binning WOE monótono y con restricciones mediante `optbinning`; los tests unitarios
verifican el IV contra un cálculo independiente. Campeón = regresión logística sobre variables WOE
(interpretable y compatible con un scorecard). Retador = LightGBM. El campeón se elige con criterios
documentados, no solo por AUC.

## ADR-006 — La etapa de PySpark corre en modo local — *Aceptada*
Contexto: el plan original era correr el pipeline de PySpark en Databricks. Eso añade una plataforma
externa que nadie que clone el repositorio puede reproducir, y el dataset (30,000 filas) no necesita
un clúster.
Decisión: el pipeline de variables en PySpark se desarrolla, prueba y ejecuta con una sesión local
de Spark (`local[*]`), con un test de paridad contra la implementación en pandas. El registro de
ejecución (versión de Spark, conteos de filas, resultado de la paridad) se guarda en `docs/evidence/`.
Consecuencias: todo el proyecto corre desde un clon limpio en una sola máquina. El README debe decir
con claridad que Spark corrió en modo local: esta etapa muestra que el pipeline es portable a Spark,
no muestra desempeño a escala. Correr el mismo código en un clúster administrado queda como trabajo
futuro opcional.

## ADR-007 — Datos crudos incluidos en el repositorio — *Aceptada*
Contexto: los resultados deben poder reproducirse desde un clon limpio sin un paso manual de
descarga, y el archivo es pequeño (alrededor de 5.5 MB). La licencia (CC BY 4.0) permite la
redistribución con atribución.
Decisión: subir el archivo crudo sin cambios, con su SHA-256 registrado en `docs/DATA.md`.

## ADR-008 — Tratamiento de las anomalías de los datos — *Aceptada*
Contexto: las verificaciones de calidad de datos (`notebooks/01_eda.ipynb`, cifras en
`reports/metrics/data_quality.json`) encontraron códigos que la documentación del dataset no
describe, además de saldos negativos, cuentas inactivas, filas idénticas y montos con colas largas.
Cada punto necesita una decisión antes de modelar. Cuando se tomaron estas decisiones todavía no
existía la partición train/test, así que se tomaron sin mirar tasas de default por código ni por
variable. El target se usó solo para el conteo global de defaults y para contar filas idénticas con
y sin la columna del target.

Decisiones:
- **`EDUCATION` 0, 5, 6 → 4 ("otros"); `MARRIAGE` 0 → 3 ("otros").** 345 filas (1.15%) y 54 filas
  (0.18%). La documentación no da significado a estos códigos, y "otros" es la única categoría
  documentada que no afirma ninguno. El mapeo está fijo en `creditrisk.data.clean`, no depende de
  los datos y falla ante cualquier código que no haya visto.
- **Los códigos −2 y 0 de `PAY_*` se conservan como categorías propias.** No se fusionan con −1 ni
  se colocan en la escala de meses de atraso. Su significado es un **SUPUESTO** (sin documentación y
  sin cita). La evidencia viene de mayo a agosto, emparejando el estado de un mes con el estado de
  cuenta de ese mes y el pago hecho el mes siguiente (ver ADR-009 para el porqué de este
  emparejamiento). Código 0: el 2.0% de las cuentas-mes no tiene estado de cuenta positivo y el 0.2%
  de los estados de cuenta positivos se paga completo, lo que encaja con "crédito revolvente".
  Código −2: el 63.4% no tiene estado de cuenta positivo, y el 93.5% de los positivos se paga
  completo, así que se lee como "sin saldo, o saldo cancelado", que es más amplio que el habitual
  "sin consumo".
- **El estado de septiembre (`PAY_1`) no está en el mismo plano que los meses anteriores.** El
  código 1 (un mes de atraso) aparece 3,688 veces en `PAY_1` y 28, 4, 2, 0 y 0 veces en
  `PAY_2`…`PAY_6`, mientras que el código 2 es frecuente en todos los meses. En septiembre, 1,689
  cuentas cuya factura es cero o negativa llevan un código de atraso; en los meses anteriores eso
  ocurre en 22 a 125 cuentas, y esas cuentas se codifican como −2. Una parte grande del grupo "un
  mes de atraso" de septiembre no es, por tanto, una mora que se pueda ver en las facturas. No se
  recodifica nada. La Fase 4 decide cómo tratan a `PAY_1` las variables que comparan meses, y la
  Fase 10 debe declararlo como limitación del análisis de roll rates: las transiciones hacia
  septiembre no se pueden leer como las de los otros meses.
- **Las facturas negativas y las que superan el límite se conservan.** Hay facturas negativas en 590
  a 688 cuentas por mes y facturas mayores que `LIMIT_BAL` en 798 a 2,115. El archivo no las define;
  se leen como saldos a favor y cuentas sobregiradas (interpretación). El piso en cero se aplica
  solo dentro de las fórmulas que lo necesitan (utilización, el proxy de EAD de ADR-004).
- **Las cuentas inactivas permanecen en la población.** 795 cuentas tienen todas las facturas y
  todos los pagos en cero durante los seis meses (866 tienen todas las facturas en cero). Tienen una
  línea de crédito abierta, así que un banco las puntuaría, y quitarlas cambiaría la población a la
  que se refiere todo resultado posterior. 495 de las 795 llevan un código de atraso en `PAY_1` sin
  haber tenido nunca nada facturado (la misma anomalía de arriba).
- **Las filas idénticas se conservan.** La columna `ID` del archivo crudo es un número de fila (de 1
  a 30,000) y no está documentada como clave de cliente; el cargador la descarta. Sin ella, 35 filas
  (0.12%) repiten una fila anterior en todas las columnas. 30 de ellas son cuentas inactivas, que
  por fuerza se parecen. Las otras 5 tienen facturas o pagos y pueden ser registros repetidos, pero
  nada en el archivo lo confirma, y eliminar filas por una conjetura cambiaría la población.
  Consecuencia para la Fase 3: 227 filas tienen un gemelo en las variables candidatas del modelo,
  así que los gemelos pueden caer a ambos lados de la partición train/test. La Fase 3 cuenta cuántos
  lo hacen y guarda el conteo.
- **No se elimina ni se trunca ningún valor.** Las facturas y los pagos tienen colas derechas largas,
  y no hay forma de distinguir un error de un saldo grande. Las variables en bins y los modelos de
  árboles no son sensibles al tamaño de un valor extremo. Los percentiles guardados en
  `data_quality.json` son solo descriptivos: cualquier umbral que necesite un paso posterior se
  ajusta ahí con datos de entrenamiento.

Consecuencias: el único cambio a los valores crudos es la recodificación de `EDUCATION` y
`MARRIAGE`. El significado de −2 y 0 sigue siendo un supuesto, y la Fase 10 debe mostrar cómo
cambian sus resultados con un mapeo distinto de estos códigos a tramos de mora.

## ADR-009 — Timing de pagos e información disponible al momento de predecir — *Aceptada*
Contexto: el archivo no dice qué estado de cuenta cancela un pago, a qué estado de cuenta y a qué
pago se refiere un estado de pago, ni en qué fecha se registró cada columna. La ingeniería de
variables necesita lo primero, y "ninguna variable usa información del período del target" necesita
lo otro.

Evidencia (`reports/metrics/data_quality.json`):
- Entre las cuentas que pagaron algo en un mes, el pago es igual al estado de cuenta del mes
  anterior en el 16.4% a 19.8% de los casos, y al del mismo mes en el 2.5% a 2.7%; casi todos estos
  últimos son cuentas cuyos dos estados de cuenta tienen el mismo monto.
- En todas las cuentas, las compras implícitas entre dos estados de cuenta consecutivos y el pago
  entre ellos son negativas en alrededor del 0.4% de las cuentas por mes si el pago cancela el
  estado de cuenta anterior, y en el 15.8% a 19.1% si cancela el del mismo mes.
- De mayo a agosto, emparejar el estado del mes k con el estado de cuenta k y el pago del mes
  siguiente encaja con los códigos documentados mucho mejor que emparejarlo con el pago del mismo
  mes: con el código −1 ("pagó puntualmente") el 98.4% de los estados de cuenta positivos se paga
  completo, frente al 75.8%.

Decisiones:
- `PAY_AMTk` se trata como el pago del estado de cuenta `BILL_AMT{k+1}`. Los ratios de pago se
  construyen como `PAY_AMTk / BILL_AMT{k+1}`, lo que da cinco meses utilizables.
- El modelo se puntúa a fines de septiembre de 2005 para predecir el default de octubre de 2005.
  Todas las columnas distintas del target son variables candidatas, excepto `SEX` y `MARRIAGE`
  (ADR-002). Que todas se conocieran antes de octubre es un **SUPUESTO** que descansa en la
  descripción del dataset.
- **`PAY_1` es la columna con más riesgo bajo ese supuesto.** Si siguiera la misma regla que los
  meses anteriores, se fijaría con el pago de octubre, que pertenece al período del resultado y no
  está en el archivo. Las anomalías de ADR-008 sugieren que en cambio se registró con otro criterio,
  pero el archivo no permite resolverlo. `PAY_1` sigue como variable candidata. La Fase 4 y la Fase 5
  deben ejecutar, solo con datos de entrenamiento, una verificación de fuga con nombre propio: su
  information value frente al umbral de alarma de 0.5, y el desempeño del modelo con y sin `PAY_1`,
  reportado como sensibilidad.
- `LIMIT_BAL`, `AGE` y `EDUCATION` tampoco tienen fecha en el archivo. Un límite de crédito puede
  ajustarse en respuesta al comportamiento, así que `LIMIT_BAL` podría reflejar una fecha posterior
  a septiembre. Esto no se puede verificar y se declara como limitación.

Consecuencias: si la verificación de `PAY_1` da la alarma, el tratamiento de `PAY_1` se decide de
nuevo en este registro antes de cerrar ningún modelo. Alternativa considerada: excluir `PAY_1` ahora
y modelar solo con los meses 2–6. Rechazada porque descartaría el comportamiento más reciente sin
evidencia de que haya fuga.

**Actualización (Fase 4): la verificación dio la alarma y la decisión cambia.** En train, el estado
de septiembre agrupado (`STATUS_SEP`) tiene un IV de 0.873, por encima del umbral de 0.5
(`reports/metrics/woe_iv.json`). El IV del estado de pago crece con la cercanía al resultado (abril
0.285, mayo 0.330, junio 0.353, julio 0.412, agosto 0.550), pero el salto de agosto a septiembre es
mucho mayor que los anteriores. Puede ser solo recencia o puede ser información de octubre; el
archivo no permite distinguirlo. Decisión: **el modelo campeón se construye sin `STATUS_SEP`**. El
modelo con `STATUS_SEP` se ajusta solo como sensibilidad, para mostrar cuánto ganaría la predicción
si el dato fuera legítimo, y se reporta como tal. Los montos de septiembre (`BILL_AMT1`, `PAY_AMT1`)
se siguen usando: en la Fase 2 quedó establecido qué estado de cuenta paga cada pago, y que se
conocieran antes de octubre sigue siendo el SUPUESTO de arriba. Consecuencia: el desempeño del
campeón será probablemente menor que el de los análisis públicos de este dataset, que suelen usar
el estado de septiembre (`PAY_0` en el archivo original); se reporta tal como salga.

## ADR-010 — Mecánica de la partición hold-out y de la validación cruzada — *Aceptada*
Contexto: ADR-003 fija el diseño (hold-out estratificado 80/20 y validación cruzada estratificada
de 5 folds con 3 repeticiones). Faltaba decidir cómo se crea y se guarda la partición para que el
conjunto de test sea siempre el mismo. Las cifras están en `reports/metrics/split.json`.

Decisiones:
- **La partición se crea una sola vez y se guarda en el repositorio**, en
  `data/splits/holdout_index.csv` (una línea por fila del archivo crudo: su posición y `train` o
  `test`). Regenerarla solo desde la semilla podría dar otro resultado con otra versión de las
  librerías; el archivo guardado fija el test. Además del archivo crudo, es el único archivo bajo
  `data/` que se sube al repositorio, y no contiene datos de clientes, solo posiciones.
  `make split` compara el archivo con lo que produce la semilla del proyecto y nunca lo reemplaza;
  si el archivo falta, da error en lugar de crear un test nuevo. Un test hace la misma comparación.
  Si esa comparación falla, el archivo guardado manda: se conserva y se investiga qué cambió
  (semilla, datos o versión de una librería).
- **Versiones de librerías.** Cada archivo de métricas guarda, junto al commit, la semilla y el
  hash de los datos, las versiones de Python y de las librerías que pueden cambiar un resultado
  (numpy, pandas, scipy, scikit-learn, statsmodels, optbinning). Las dependencias solo fijan una
  versión mínima; así, si un clon futuro obtiene otro número, se puede ver con qué versiones se
  produjo el original.
- **Las filas se identifican por su posición en el archivo crudo** (0 a 29,999). La columna `ID` se
  descarta al cargar (ADR-008), y la posición es estable porque el hash del archivo está fijado.
- **Se estratifica solo por el target.** Train: 24,000 filas y 5,309 defaults; test: 6,000 filas y
  1,327 defaults. La tasa de default es 22.12% en ambas partes, con una diferencia de 0.004 puntos
  porcentuales. Esto se cumple por construcción: la estratificación reparte los defaults en
  proporción. Como referencia de la precisión con que 6,000 filas estiman una tasa, el intervalo de
  Wilson del 95% para la tasa de test va de 21.1% a 23.2%.
- **Los folds de validación cruzada no se guardan en archivo, pero quedan fijados por una huella.**
  Se generan con la semilla del proyecto sobre train, en su orden original
  (`creditrisk.data.split.make_cv`). Los 15 folds de validación tienen 4,800 filas cada uno, con
  1,061 o 1,062 defaults. Su huella SHA-256 está guardada en `split.json`, y un test la compara con
  los folds que se generan hoy: si una actualización de librerías los cambiara, el test fallaría en
  vez de cambiar en silencio todos los resultados de validación cruzada.
- **Los gemelos no se mueven de lado.** 43 de las 6,000 filas de test tienen un gemelo en train en
  las variables candidatas del modelo, y 41 de esas 43 son cuentas inactivas (todas las facturas y
  pagos en cero). 42 filas de train tienen un gemelo en test, y 140 filas de train tienen un gemelo
  dentro de train, que puede caer en otro fold. Forzar que queden del mismo lado supondría que son
  el mismo cliente, cosa que el archivo no permite afirmar (ADR-008). El efecto posible es que el
  test y la validación cruzada se vean ligeramente optimistas.
- **Sensibilidad por gemelos, fijada antes de ver el test.** Cada evaluación final sobre el test se
  reporta sobre las 6,000 filas y, como sensibilidad, sobre las 5,957 filas que no tienen un gemelo
  en train. Se decide ahora para que no dependa de ningún resultado de test.
- **Acceso al test.** El código de las fases siguientes debe usar `load_train()` para todo.
  `load_test()` se reserva para la evaluación final de un modelo terminado: una vez por modelo, sin
  cambiar nada después por lo que muestre. Un test revisa el código y los notebooks y falla si un
  archivo que no está en una lista corta carga el test o el archivo completo.

Limitación: train y test vienen del mismo período (abril a septiembre de 2005, resultado en
octubre), así que el test mide el desempeño dentro de la misma foto y no hacia el futuro. No
sustituye a una validación fuera de tiempo, que estos datos no permiten (ADR-003). Además, las 43
filas de test con gemelo en train no son casos de prueba del todo independientes.

## ADR-011 — Variables candidatas, binning WOE y preselección — *Aceptada*
Contexto: la Fase 4 construye las variables, las agrupa en bins con WOE y decide cuáles pasan al
modelo. Todo se calcula con la parte de entrenamiento (24,000 filas); las cifras están en
`reports/metrics/woe_iv.json` y el análisis en `notebooks/02_woe_iv_binning.ipynb`.

Decisiones tomadas antes de calcular:
- **Variables.** 24 candidatas construidas fila por fila, sin ajustar nada a los datos
  (`creditrisk.features.engineering`): límite, edad y educación; utilización por mes (la factura
  negativa cuenta como cero), su media, su máximo y su tendencia (septiembre menos abril); ratio de
  pago `PAY_AMTk / BILL_AMT{k+1}` por mes y su media (vacío cuando no había nada que pagar);
  historial de mora de agosto a abril (meses con atraso, peor atraso, meses desde el último atraso);
  estado de septiembre y de agosto como categorías; e indicador de cuenta inactiva.
- **El historial de mora usa solo agosto a abril**, los meses con codificación consistente
  (ADR-008). Así la sensibilidad con y sin `STATUS_SEP` (ADR-009) quita toda la información del
  **estado** de septiembre y no deja parte escondida dentro de los conteos. Los montos de
  septiembre siguen en otras variables (`UTIL_1`, `UTIL_MEAN`, `UTIL_MAX`, `UTIL_TREND` y
  `PAY_RATIO_1`).
- **Códigos de estado como categorías.** −2, −1 y 0 son categorías propias, no "cero meses de
  atraso" (ADR-008). Los atrasos de 3 o más meses se agrupan en "3+": es una regla fija de negocio,
  que no mira el target, y evita que un código casi vacío (el 8 tiene muy pocos casos) se una a una
  categoría sin relación.
- **Binning.** Cortes encontrados con `optbinning` (ADR-005); cada bin tiene al menos el 5% de las
  filas, salvo el bin de valores vacíos, que no tiene mínimo. Para las variables numéricas se
  declaró, según la lógica de negocio, la dirección de la tasa de default: sube con la utilización
  y con la mora, baja con el ratio de pago, el límite y los meses desde el último atraso; la edad va
  sin restricción. La asignación a bins, el WOE y el IV se calculan en `creditrisk.features.woe`, y
  un test comprueba que ese IV coincide con el de `optbinning` en los mismos bins.
- **Preselección.** Se recorre la lista de mayor a menor IV y se conserva cada variable salvo que su
  IV sea menor que 0.02 o que su WOE tenga una correlación mayor que 0.7 con una ya conservada. Un
  VIF mayor que 5 se marca. La lista resultante es descriptiva: la Fase 5 repite binning,
  preselección y selección dentro de cada fold, porque el IV calculado una vez sobre todo train
  vio todas las filas.

Resultados y decisiones tomadas con ellos a la vista:
- **Cinco variables superan el IV de alarma de 0.5:** `STATUS_SEP` (0.873, ver ADR-009),
  `N_DELAY_MONTHS` (0.641), `MONTHS_SINCE_DELAY` (0.630), `STATUS_AUG` (0.550) y `MAX_DELAY`
  (0.531). Las cuatro últimas vienen de agosto a abril, meses anteriores a septiembre y con
  codificación consistente, así que no deberían contener información de octubre, bajo el SUPUESTO
  de ADR-009 sobre las fechas de registro. Lo apoyan el crecimiento gradual del IV mensual de abril
  a agosto y que, en los 15 folds, el IV fuera del fold de `STATUS_AUG` (0.47 a 0.62) no se cruza
  con el de `STATUS_SEP` (0.78 a 0.97). **Se conservan.** `N_DELAY_MONTHS`, `MAX_DELAY` y
  `MONTHS_SINCE_DELAY` mantienen el orden de riesgo de sus bins en los 15 folds; en las dos últimas
  eso dice poco, porque el binning les dejó solo dos bins (en `MAX_DELAY`, "algún atraso" frente a
  "ninguno", sin la severidad). El orden de `STATUS_AUG` es menos estable (correlación de rangos
  media 0.73 y mínima 0.40) porque tres de sus categorías tienen un riesgo muy parecido.
- **`AGE` y `EDUCATION` quedan fuera del modelo** (ADR-002). Su IV en train es 0.022 en ambos
  casos, en el límite inferior de lo útil. Con muestras chicas el IV fuera del fold se infla, más
  cuantos más bins hay, así que se midió cuánto daría el azar barajando el target de validación 100
  veces por fold con los mismos bins:
  - `AGE`: su IV fuera del fold (0.030 de media) es en buena parte azar (0.015 con sus 14 bins);
    solo supera al azar en 10 de 15 folds, y el orden de riesgo de sus bins no se sostiene
    (correlación de rangos media 0.66 y mínima 0.27). **Su señal no se distingue bien del azar.**
  - `EDUCATION`: con 3 bins supera al azar en los 15 folds, así que su señal es real, aunque débil
    (llega a un IV de 0.02 fuera del fold en 10 de 15 folds porque está justo en ese umbral). **Una
    señal tan débil no alcanza para justificar un atributo demográfico**, como exige ADR-002.

  Historia de la decisión: el criterio de estabilidad se precisó al ver los primeros resultados de
  train, y la línea de azar se agregó después, al revisar la fase. El responsable del proyecto
  decidió la exclusión de ambas y, con la línea de azar a la vista, confirmó la de `EDUCATION`
  corrigiendo el motivo (al principio se había registrado como "inestable"). Ambas variables se
  incluyen en la auditoría de equidad de la Fase 12.
- **`INACTIVE` no aporta información:** las cuentas inactivas son 646 de las 24,000 de train (2.7%),
  menos que el tamaño mínimo de un bin, así que la variable queda en un solo bin. Esas cuentas
  quedan incluidas, junto con otras, en los bins de "nada que pagar" de los ratios de pago.
- **`STATUS_AUG`:** el binning unió el código 1 (muy pocos casos en agosto) con el −2, que no tiene
  relación de negocio. No afecta a nada porque la variable no queda preseleccionada, pero habría que
  corregirlo antes de reutilizarla (por ejemplo en la Fase 10).
- **Lista preseleccionada (10 variables):** `N_DELAY_MONTHS`, `PAY_RATIO_1` a `PAY_RATIO_5`,
  `PAY_RATIO_MEAN`, `LIMIT_BAL`, `UTIL_6` y `UTIL_2`. `N_DELAY_MONTHS` deja fuera a `MAX_DELAY`,
  `MONTHS_SINCE_DELAY` y `STATUS_AUG` por su correlación de WOE con ella (0.920, 0.915 y 0.763). El
  VIF máximo entre las diez es 2.55. El modelo de sensibilidad de ADR-009 parte de las mismas
  candidatas más `STATUS_SEP`, y en la Fase 5 la preselección se repite dentro de cada fold.

Limitaciones:
- El IV y la preselección son descriptivos (calculados una vez con todo train).
- Las decisiones sobre `AGE` y `EDUCATION` se tomaron con los 15 folds que la Fase 5 usará para
  medir el modelo. Como solo quitaron dos variables con IV cercano a 0.02, el optimismo posible es
  mínimo, y se prefirió mantener un único esquema de folds.
- 140 filas de train tienen un gemelo en las variables candidatas (ADR-010) y pueden caer en folds
  distintos, lo que favorece un poco la estabilidad medida.
- Las medidas de estabilidad son gruesas: la correlación de rangos con pocos bins dice poco, y el
  conteo de folds con IV ≥ 0.02 es poco informativo para variables con muchos bins.
- La única variable de tendencia (utilización de septiembre menos la de abril) resultó inútil (IV
  0.002); no se probaron tendencias del ratio de pago ni de la mora.
- `LIMIT_BAL` no tiene fecha y un banco puede ajustarlo según el comportamiento (ADR-009).

## ADR-012 — Especificación, selección y validación del modelo de PD — *Aceptada*
Contexto: la Fase 5 ajusta la regresión logística sobre las variables WOE de ADR-011 y mide su
discriminación y su calibración. Las cifras de entrenamiento están en
`reports/metrics/pd_logistic_cv.json` y el análisis en `notebooks/03_pd_model_logistic.ipynb`.

Decisiones tomadas antes de ajustar:
- **Validación cruzada de todo el procedimiento.** El binning, la preselección, la selección de
  variables y la logística forman una sola cadena (`creditrisk.models.pipeline.WoeLogitModel`), que
  se vuelve a ajustar dentro de cada uno de los 15 folds de ADR-010 y se mide en las filas que ese
  fold no vio. Un test comprueba que el binning de cada fold no ve sus filas de validación (los
  pasos siguientes usan las mismas filas), y otro que las predicciones fuera de fold quedan en las
  filas correctas. El modelo final se ajusta después con todo train.
- **Selección dentro de la logística.** Con WOE = ln(no-defaults / defaults), todo coeficiente de un
  modelo de PD debe ser negativo. Se quita primero, de a una, cualquier variable con signo positivo
  (la de mayor p-valor entre ellas) y luego, de a una, la de mayor p-valor si supera 0.05.
- **Sin regularización ni recalibración.** Máxima verosimilitud simple con statsmodels, que da
  errores estándar e intervalos de confianza interpretables. La calibración se reporta tal como sale.
- **Medición.** AUC, Gini, KS y PR-AUC sobre las predicciones fuera de fold, con intervalo bootstrap
  estratificado de 1,000 remuestreos (primera repetición, semilla del proyecto), más el AUC de cada
  repetición y el rango entre folds. Calibración: Brier, calibración global (calibration-in-the-large,
  ideal 0), pendiente (ideal 1) con su intercepto, y tabla por decil. Puntos de operación: marcar el
  10%, 20% o 30% de mayor PD, o todo lo que supere la tasa de default de train o 0.5.
- **Modelo de sensibilidad (ADR-009).** El mismo procedimiento con `STATUS_SEP` agregado a las
  candidatas dentro de cada fold, no a una lista fija.
- **Mismos folds que la Fase 4.** Esos folds ya sirvieron para excluir `AGE` y `EDUCATION`
  (ADR-011); se mantienen para tener un solo esquema y se declara como limitación.
- **Uso del test.** Una sola evaluación, con los dos modelos congelados: el archivo de métricas de
  entrenamiento guarda las variables y los coeficientes de cada modelo, y
  `creditrisk.models.final_evaluation` se niega a evaluar un modelo que no coincida con ellos, o una
  segunda versión de un modelo si el test ya se usó. Es el único módulo de la librería, además de
  `creditrisk.data.split`, que carga el test, y solo puede llamarse desde este notebook (lo
  comprueba un test). Se reporta sobre las 6,000 filas y sobre las que no tienen gemelo en train
  (ADR-010), con la diferencia de AUC entre ambos modelos por bootstrap pareado en los dos casos.
  Los puntos de operación en test son los umbrales de PD fijados en train (la tasa de default, 0.5
  y los umbrales que marcaron el 10%, 20% y 30% en validación cruzada) más la captura al marcar el
  10%, 20% o 30% de las cuentas de test con mayor PD, que es una regla de capacidad. El resultado
  se guarda en `reports/metrics/pd_logistic_test.json`.

Resultados en train:
- **Campeón: 5 variables**, todas con signo negativo y p-valor menor que 0.01: `N_DELAY_MONTHS`,
  `PAY_RATIO_1`, `PAY_RATIO_MEAN`, `LIMIT_BAL` y `PAY_RATIO_4`. La logística quitó, en este orden,
  `UTIL_6`, `PAY_RATIO_5` y `UTIL_2` por signo incorrecto, y `PAY_RATIO_3` y `PAY_RATIO_2` por
  p-valor. Las dos variables de utilización solo invierten su signo una vez que el modelo conoce el
  límite y el comportamiento de pago (las demás de utilización ya habían salido por correlación con
  ellas); `PAY_RATIO_5` se invierte porque se superpone con los otros ratios del modelo.
- **Discriminación fuera de fold del campeón:** AUC 0.718 (IC 95% 0.710 a 0.725), KS 0.341
  (0.329 a 0.357), PR-AUC 0.459 (0.447 a 0.471); el AUC por fold va de 0.707 a 0.738 y en el
  propio train es 0.720, así que no hay señales de sobreajuste. El intervalo remuestrea filas con
  los modelos de cada fold fijos; el rango entre folds y entre repeticiones refleja el reajuste.
- **Calibración del campeón:** calibración global −0.001 y pendiente 0.986; la PD media predicha
  coincide con la tasa observada. Por decil, la PD sigue la tasa observada en conjunto, pero **la
  tasa observada no crece de forma ordenada en los deciles bajos**: el decil 2 tiene una tasa mayor
  que el 3 y que el 4, y el 5 una apenas mayor que el 6. La mayor diferencia está en el decil 2
  (PD 11.0% frente a 12.7% observada). El modelo separa bien a las cuentas de mayor riesgo y poco
  entre las de riesgo bajo y medio. No se cambia el modelo por ello.
- **Modelo de sensibilidad:** AUC fuera de fold 0.761 (IC 95% 0.753 a 0.768), y sí ordena todos sus
  deciles. Si el estado de septiembre fuera legítimo, esa diferencia sería lo que cuesta excluirlo;
  si no lo es, no es un costo. El intervalo pareado de la diferencia está en los resultados en test.
- **Selección entre folds:** en el campeón, `LIMIT_BAL` y `PAY_RATIO_MEAN` entran en los 15 folds y
  `N_DELAY_MONTHS` y `PAY_RATIO_1` en 14; `PAY_RATIO_2` a `PAY_RATIO_5` se reemplazan entre sí. La
  lista exacta de variables no es única, aunque el desempeño sí es estable.

Resultados en test (una sola evaluación, `reports/metrics/pd_logistic_test.json`):
- **Campeón:** AUC 0.719 (IC 95% 0.703 a 0.736), KS 0.359 (0.333 a 0.390), PR-AUC 0.474 (0.449 a
  0.502); sin las 43 filas con gemelo en train, AUC 0.721 (0.705 a 0.737). Es compatible con la
  validación cruzada (0.718): no se detecta optimismo, aunque un solo test no permite descartar uno
  menor que su incertidumbre (unos ±0.017). PD media 21.8% frente a 22.1% observada; calibración
  global 0.024 y pendiente 1.037. También en test la tasa observada no crece de forma ordenada en
  los deciles bajos y medios (el decil 3 por debajo del 2; el 6 por debajo del 4 y del 5).
- **Modelo de sensibilidad:** AUC 0.764 (0.748 a 0.779). Diferencia con el campeón 0.044 (IC 95%
  pareado 0.035 a 0.054; 0.043 sin gemelos, 0.033 a 0.053).
- **Nada se cambia por estos resultados:** el campeón sigue siendo el modelo sin el estado de
  septiembre (ADR-009).

Limitaciones: una sola foto temporal (ADR-003); los folds ya se usaron en la Fase 4, con un efecto no
cuantificado en validación cruzada (en test no se observó optimismo); 140 filas de train con gemelo
pueden caer en folds distintos (ADR-010); los errores estándar de los coeficientes no consideran que
las variables y los bins se eligieron con los mismos datos, y son algo optimistas; la calibración
depende de la tasa de default de este archivo, que no se puede leer como la de la cartera del banco
(`docs/DATA.md`); las fechas de registro de las variables son un SUPUESTO (ADR-009).
