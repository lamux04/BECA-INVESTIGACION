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

= Análisis de los problemas del desbalanceo de clases en Predective Mutation Testing

== INTRODUCTION

El *software testing* es una fase crucial en el ciclo de vida de desarrollo de software y tiene como objetivo identificar y eliminar fallos para asegurar la alta calidad de los productos software.

El *mutation testing* es una técnica para evaluar la efectividad de los tests introduciendo pequeñas modificaciones, o mutaciones al código del programa determinando si esos cambios son detectados o no por los tests.

El *Predictive Mutation Testing (PMT)* usa Machine Learning para predecir el resultado de un mutante sin tener que ejecutar toda la suite de tests.

La efectividad de los modelos de PMT está muy influencia por factores como el preprocesamiento de datos, el desbalanceo de clases y la selección de modelo. Estos datsets tienen un alto desbalanceo y sesga los modelos hacia la clase mayoritaria.

#importante[
Este problema es el mismo que nos encontramos con los datasets de ciberseguridad, la idea de mi TFG sería usar técnicas de preprocesamiento y filtrado que se usa para PMT pero usarlo en los datasets de ciberseguridad.
]

Para intentar arreglar este desbalanceo, el paper usa técnicas como *RandomUnderSampler (RUS)*, *Synthetic Minority Over-sampling Technique (SMOTE)* y *Edited Nearest Neighbors (ENN)*.

== MÉTODOS

=== Preprocesamiento de datos

Para hacer un filtrado de los datos, elimina los _unreached mutants_ que son muestras triviales en las que claramente, al no ser probados por ningun test, pasan los tests correctamente.

#importante[
Para el caso de los datasets de ciberseguridad, habría que buscar muestras triviales (como los unreached mutants) que podamos eliminar para reducir el tamaño de los datasets.
]

Lo siguiente que hace es dividir los datos en 80% de entrenamiento y 20% de test. Esta división se realiza por proyectos en lugar de por mutantes. Esto se hace para que en los tests no aparezcan mutantes de proyectos que estan en el entrenamiento.

#importante[
En el caso de los datasets de ciberseguridad, la división sería simplemente un 80% de entrenamiento y 20% de test. No tenemos el problema de los proyectos y mutantes.
]

Después, realiza una normalización *RobustScaler*. Este tipo de normalización usa la mediana y el rango intercuartílico en lugar de la media y varianza. Esto permite que la normalización sea más robusta y menos sensible a los extremos.

$
x' = frac(x - "mediana", "IQR")\
"IQR" = Q_3 - Q_1
$ 

*_IMPORTANTE_*: La normalización de los datos se realiza únicamente con el cálculo de la mediana y el IQR a partir de los datos de entrenamiento.

#importante[
En el caso de los datasets de ciberseguridad, la normalizaicón se realizaría de la misma forma. Recalcar que para calcular la mediana y el rango intercuartílico, sólo debemos usar los datos de entrenamiento.
]

=== Desbalanceo de clases

Para abordar el problema del desbalanceo de clases, el paper usa 3 técnicas de rebalanceo:

- *RandomUnderSampler (RUS)*: Elimina instancias aleatoriamente de la clase mayoritaria produciendo una distribución de 50%.
- *Synthetic Minority Nearest Neighbours (SMOTE)*: Genera ejemplos sintéticos de la clase minoritaria interpolando entre clases existentes. Esto logra un una distribución del 50% también.
- *Edited Nearest Neighbours (ENN)*: Elimina instancias ambiguas o mal etiquetadas de la clase mayoritaria. Esto no logra una distribución del 50% pero lo acerca.

#importante[
En el caso de los datasets de ciberseguridad, podría probar las técnicas de RUS, SMOTE y ENN. También se me ocurre que se podría hace un primer proceso de ENN y después otro proceso de RUS o SMOTE para terminar de realizar el balanceo. Serían en total 5 posibilidades:
- RUS.
- SMOTE.
- ENN.
- ENN + RUS.
- ENN + SMOTE.
]

=== Métricas de evaluación

Para la evaluación de los modelos usa *Precision*, *Recall*, *F1 Score*, *Matthews Correlation Coefficient (MCC)* y *Area Under the Receiver Operating Characteristic Curve (AUCROC)*.

#image("assets/image-1.png")

#importante[
Esto se ve mas a fondo en otro paper.
]