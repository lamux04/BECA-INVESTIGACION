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

= Detección de ciberataques basado en LLM

== Introducción

El *Transfer Learning* consiste en modificar parte de la arquitectura de una ANN existente que ya ha sido entrenada, reemplazando las últimas capas con nuevas que sean relevantes para la nueva tarea y entonces, reentrenar sólo las capas alteradas para adaptar un modelo al nuevo dominio.

El *Fine-Tunning* consiste en modificar los pesos de una ANN existente sin alterar su arquitectur para resolver una tarea específica.

#importante[
  En este paper, y en mi TFG, se usa la técnica de _Fine-Tunning_ de un encoder-decoder ya preentrenado, como es el modelo T5.
]

== Datasets de ciberseguridad

En mi TFG voy a usar 3 datasets de ciberseguridad:

- *CIC-IDS-2017*: Incluye tráfico benigno y 14 tipos de ataques agrupados en 7 tipos principales. El entorno de este dataset es lo mas realista posible. Este dataset tiene 84 características.
- *CSE-CIC-IDS2018*: Sigue la misma estructura que el anterior.
- *BCCC-CIC-IDS-2017*: Es un dataset aumentado. Tiene 122 características.

#table(
  columns: 5,
  align: (left, center, center, center, center),
  stroke: 0.5pt,
  inset: 6pt,

  [*Dataset*], [*Year*], [*Attack Types*], [*Features*], [*Records*],

  [CIC-IDS-2017], [2017], [14], [84], [2,830,743],
  [CSE-CIC-IDS2018], [2018], [14], [84], [16,232,943],
  [BCCC-CIC-IDS-2017], [2025], [13], [122], [2,438,052],
)

== Preprocesamiento de datos

=== Limpieza de datos

+ Homogenización de nombres de columnas.
+ Eliminación de columnas inútiles. Son columnas que usan sólo muy pocos patrones. Estas columnas son: `FLOW ID`, `SRC_IP`, `SRC_PORT` y `DST_IP`.
+ Eliminación de columnas con valores únicos. Eliminamos las columnas con desviación estandar $0$ o $0.01$.
+ Eliminación de columnas con alta correlación. 
  + Calculamos la matriz de correlación.
  + Obtener la mátriz triangular superior.
  + Listar las coumnas cuya correlación sea mayor a $0.95$.
  + Eliminar las columnas con alta correlación.
+ Eliminar filas con valores infinito, vacío o nulo.
+ Eliminación de filas duplicadas.

#importante[Aquí iria el filtrado o todo lo nuevo que quisiera hacer yo.]

=== Transformación de datos

+ Preparación de las cadenas de entrada. Se transforma todas las características de entrada en un único string separados con `|`.
+ Etiquetado: Codificamos las etiquetas usando números secuenciales, empezando con $0$.

=== Undersampling

El dataset, después de hacer la limpieza se encuentra muy desbalanceado. Por ello, para balancearlo hacemos un undersampling de la clase mayoritaria (benigna) y lo balanceamos a 50% beningo y 50% maligno. Aun así, las clases malignas siguen desbalanceadas.



