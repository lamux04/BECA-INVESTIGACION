// Fuente principal (sans-serif)
#set text(
  font: "Roboto",
  size: 11pt
)

// Interlineado y separación equilibrados
#set par(
  leading: 0.7em,
  spacing: 1.8em
)

// Estilo de títulos
#show heading.where(level: 1): it => [
  #text(
    fill: rgb("#1E3A8A"), // azul profesional
    weight: "bold",
    size: 22pt,
  )[ #it.body ]
]

#show heading.where(level: 2): it => [
  #text(
    fill: rgb("#028306"),
    weight: "bold",
    size: 16pt,
  )[ #it.body ]
]

#show heading.where(level: 3): it => [
  #text(
    fill: rgb("#6a6a00"),
    weight: "semibold",
    size: 13pt,
  )[ #it.body ]
]

#show heading.where(level: 4): it => [
  #text(
    fill: rgb("#4f006f"),
    weight: "semibold",
    size: 13pt,
  )[ #it.body ]
]

#show strong: it => text(
  weight: "bold",
  fill: rgb("#763b01"),
)[#it]

#set enum(numbering: "1.", indent: 2em)
#set list(marker: "-", indent: 2em)

#let importante(body) = box(
  fill: rgb("#eef7ff"),
  stroke: 1pt + rgb("#4a90e2"),
  inset: 10pt,
  radius: 6pt,
)[
  #body
]

= Métricas de evaluación y tests estadísticos para machine learning

== Clasificación binaria

La *clasificación binaria* consiste en que un dato puede ser positivo o negativo. Esto provoca que cuando hagamos una predicción tengamos 4 posibilidades diferentes y de ahí sale la matriz de confusión:

#table(
  columns: 3,
  align: center,
  stroke: 1pt,
  [], [*Real Positivo*], [*Real Negativo*],
  [*Predicción Positivo*], [TP], [FP],
  [*Predicción Negativo*], [FN], [TN],
)

=== Accuracy

$
  "Acc" = frac("TP" + "TN", "TP" + "TN" + "FP" + "FN")
$

Esta medida calcula la proporción total de aciertos. El problema es que en datasets desbalanceados esta métrica puede engañar.

=== Sensitivity o recall

$
  "Sen" = frac("TP", "TP" + "FN")
$

Esta medida calcula de los positivos reales cuántos detecto.

=== Specificity (TNR)

$
  "Spe" = frac("TN", "TN" + "FP")
$

Esta medida calcula de los negativos reales cuántos clasifico bien.

=== Precision

$
  "Pre" = frac("TP", "TP" + "FP")
$

Esta medida calcula, de los que predigo como positivo, cuántos son realmente positivos.

=== F1-score

$
  "F1" = frac(2 · "Pre" · "Rec", "Pre" + "Rec")
$

Es la media armónica de la precisión y el recall. Esta medida sirve  cuando hay desbalanceo e importa el equilibrio entre detectar y no equivocarse.

=== Youden's index

$
  "Youden" = "Sensitivity" + "Specificity" - 1
$

Esta medida mide el equilibrio entre ambas clases. Va entre 0 y 1:

- 0 significa clasificador aleatorio.
- 1 significa clasificador perfecto.

=== Cohen's Kappa (K)

$
  K = frac("Acc" - p_e, 1 - p_e)
$

donde $p_e$ es la precisión esperada por azar.

Esta medida mide cuánto mejora la precisión respecto al azar.

- 1 significa perfecto.
- 0 significa azar.
- -1 significa peor que azar.

=== MCC (Matthew's correlation coefficient)

$
  "MCC" = frac("TN" · "TP" - "FN" · "FP", sqrt(("TP" + "FP")("TP" + "FN")("TN" + "FP")("TN" + "FN")))
$

Esta medida calcula la correlación entre la predicción y la realidad. Va de -1 a 1:

- 1: perfecto.
- 0: azar.
- -1: completamente incorrecto.

Esta es la mejor métrica cuando hay desbalanceo.

=== ROC Curve

Esta métrica es una gráfica.

#image("assets/image-2.png")

En el eje Y ponemos el sensitivity y en el eje X el specificity. Cada punto de la curva tiene un umbral diferente. Si la curva está cerca de la esquina superiori izquierda la curva es buena.

=== AOC (Area Under Curve)

Se calcula como el área bajo la curva ROC.

Va de 0 a 1:
- 0.5: Azar.
- 0.8: Bueno.
- 0.9+: Excelente.


=== Cross-Entropy Loss

$
  H(p, q) = -sum p_i log(q_i)
$

Mide la diferencia entre probabilidades predichas y reales. Es una métrica de optimización.

== Clasificación multiclase

La *clasificación multiclase* aparece cuando el modelo debe asignar cada instancia a una de $k$ clases, con $k > 3$. A diferencia de la clasificación binaria, donde solo hay dos etiquetas posibles, aquí el objetivo es escoger *una única clase* entre varias (por ejemplo: *negative*, *COVID-19*, *pneumonia* y *tuberculosis*).

=== Matriz de confusión $k times k$

Los resultados del clasificador se representan mediante una *matriz de confusión* de tamaño $k times k$. Cada elemento $n_(i j)$ indica cuántas instancias cuya clase real es $i$ fueron clasificadas como $j$.

- Los elementos de la diagonal $n_(i i)$ representan aciertos.
- Los elementos fuera de la diagonal representan errores (confusiones entre clases).

=== Reducir multiclase a binario (one-vs-all)

Para calcular métricas como *precision*, *recall* o *F1* (definidas originalmente para binario), se suele transformar el problema en $k$ problemas binarios *one-vs-all*.

Para cada clase $i$:

- $"TP"_i = n_(i i)$  
- $"FN"_i = sum_(j != i) n_(i j)$  (instancias reales de $i$ mal clasificadas)
- $"FP"_i = sum_(j != i) n_(j i)$  (instancias de otras clases clasificadas como $i$)
- $"TN"_i = sum_(j != i) sum_(h != i) n_(j h)$  (todo lo demás)

Con estos valores, se pueden calcular para cada clase $i$ las métricas de binario: *precision*, *recall*, *specificity*, *F1*, etc.

=== Macro-averaging

En *macro-averaging* se calcula la métrica por separado para cada clase y luego se promedia:

$
  "Macro-Métrica" = frac(1, k) sum_(i=1)^k "Métrica"_i
$

Esto asigna *el mismo peso a cada clase*, independientemente de cuántas instancias tenga.  
Es especialmente útil cuando hay *desbalance* entre clases, porque evita que las clases grandes dominen el resultado.

=== Micro-averaging

En *micro-averaging* primero se suman globalmente los conteos de todas las clases y luego se calcula la métrica una sola vez:

$
  "TP" = sum_(i=1)^k "TP"_i,quad
  "FP" = sum_(i=1)^k "FP"_i,quad
  "FN" = sum_(i=1)^k "FN"_i,quad
  "TN" = sum_(i=1)^k "TN"_i
$

Luego se aplica la fórmula de la métrica (por ejemplo precision o recall) usando esos totales.

Este enfoque da *el mismo peso a cada instancia*, por lo que puede verse dominado por las clases más numerosas.

=== Accuracy en multiclase

La *accuracy* en multiclase se calcula como el número total de aciertos (diagonal) dividido entre el total de instancias:

$
  "Acc" = frac(sum_(i=1)^k n_(i i), n)
$

=== Cohen's Kappa ($kappa$) en multiclase

Cohen's $kappa$ mide cuánto mejor es el clasificador que el azar.

$
  kappa = frac(p_0 - p_e, 1 - p_e)
$

donde:

$
  p_0 = frac(1, n) sum_(i=1)^k n_(i i)
$

es la precisión observada, y

$
  p_e = frac(1, n^2) sum_(i=1)^k n_(i dot) n_(dot i)
$

es la precisión esperada por azar (usando los totales por fila y por columna).

=== MCC en multiclase

El *Matthews Correlation Coefficient* también tiene versión multiclase:

$
  "MCC" =
  frac(
    n sum_(i=1)^k n_(i i) - sum_(i=1)^k n_(i dot) n_(dot i),
    sqrt(
      (n^2 - sum_(i=1)^k n_(i dot)^2)
      (n^2 - sum_(i=1)^k n_(dot i)^2)
    )
  )
$

Al igual que en binario, el MCC toma valores en $[-1, 1]$:

- $1$: clasificación perfecta
- $0$: comportamiento aleatorio
- $-1$: clasificación completamente opuesta

En el caso especial $k = 2$, tanto $kappa$ como MCC se reducen a sus fórmulas de clasificación binaria.