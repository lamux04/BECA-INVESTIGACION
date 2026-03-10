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

= Técnicas de undersampling

Las técnicas de undersampling se pueden agrupar en 2 grupos: métodos de generación de prototipos y métodos de selección de prototipos.

#importante[
  Para mi TFG, usaría 3:
  - RandomUnderSampler.
  - NearMiss.
  - TomekLinks
]

== Generación de prototipos

Este grupo de métodos consiste en generar prototipos de un grupo de patrones y esos protipos serán los q usaremos.

=== `ClusterCentroids`

Este método usa los k-means para encontrar los prototipos.

== Selección de prototipos

Este grupo de métodos seleccionan instancias del dataset original, no las genera. Esto tiene a su vez dos grupos: técnicas de under-sampling controlado y técnicas de limpieza de under-sampling.

=== Técnicas de under-sampling controlado

Reduce el número de observaciones de la clase objetivo a un número especificado por el usuario.

==== `RandomUnderSampler`

Selecciona aleatoriamente un subconjunto de datos de la clase objetivo.

==== `NearMiss`

Usa algunas reglas heurísticas para seleccionar las instancias. Hay 3 reglas heurísticas posibles.

=== Técnicas de limpieza de under-sampling

Estos métodos eliminan instancias ruidosas que son demasiado fáciles de clasificar.

==== `TomkeLinks`

Un Tomek's link existe cuando dos instancias de diferente clase son los vecinos más cercanos de cada uno.

Este método detecta y elimina Tomek¡s links.

==== `EditedNearestNeighbours`

Consiste en usar los k vecinos más cercanos para identificar los vecinos de las instancias de clases objetivo y eliminar esas observaciones si alguno o la mayoría de sus vecinos son de una clase diferente.

==== `RepeatedEditedNearestNeighbours`

Repite el algoritmo anterior múltiples veces.

==== `AllKNN`

Similar al anterior pero el número de vecinos evaluados en cada ronda incrementa.

==== `CondensedNearestNeighbour`

Usa el vecino más cercano para decidir iterativamente si una instancia debería de ser eliminada.

==== `OneSidedSelection`

Elimina primero las observaciones difíciles de clasificar y después usará TomekLinks para eliminar instancias ruidosas.