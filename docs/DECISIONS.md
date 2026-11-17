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
