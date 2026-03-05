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

= Convención de identificadores

== Numeración de datasets

La identificación de los datasets es de la siguiente forma:

#importante[
 {DatasetBase}\_{Transformaciones}\_{Version}
]

=== DatasetBase

Se referiere al dataset base del que parte este nuevo dataset. Puede ser uno de los siguientes:

#table(
  columns: 2,
  align: (center, center),
  inset: 10pt,

  [*Dataset*], [*DatasetID*],

  [CIC-IDS-2017], [CIC17],
  [CSE-CIC-IDS2018], [CIC18],
  [BCCC-CIC-IDS-2017], [BCCC17],
)

=== Transformación

Se refiere a la lista de transformaciones que hemos aplicado para pasar del dataset base al dataset actual. Se separan por \_.

Las transformaciones pueden ser las siguientes:

#table(
  columns: 2,
  align: (center, center),
  inset: 10pt,

  [*Transformación*], [*Transformación ID*],
)

=== Versión

Se refiere a la versión del dataset. La versión cambia sólo si se corrige algo de la técnica o se corrige algo de dentro pero la técnica sigue siendo la misma. Puede ser v1, v2, ...

== Numeración de modelos

La identificación de los modelos es de la siguiente forma:

#importante[
  {DatasetID}\_\_{ModeloBase}\_\_{Config}\_\_{Fold}\_\_{Version}
]

=== DatasetID

El dataset es el nombre completo del dataset (no solo el dataset base).

=== ModeloBase

Esto consiste el modelo base del que se parte para hacer el fine tunning. En nuestro caso normalmente será T5_small. Aquí está la tabla con todos los identificadores:

#table(
  columns: 2,
  align: (center, center),
  inset: 10pt,

  [*Modelo*], [*ID del modelo*],

  [T5 small], [T5small],
)

=== Config

Esto consiste en un identificador que identifique de un vistazo la configuración de hipeparámetros más importante que se ha usado. Aquí una tabla con los identificadores:

#table(
  columns: 2,
  align: (center, center),
  inset: 10pt,

  [*Configuración*], [*ID de la configuración*],
)

== Identificación de scripts

Para identificar los scripts usamos lo siguiente:

#importante[
  {TipoFuncion}\_{Tecnica}\_vX.py
]

=== TipoFuncion

Esto se refiere a la fase del pipeline en la que se encuentra el script. Las fases posibles están en la siguiente tabla con sus identificadores:

#table(
  columns: 2,
  align: (center, center),
  inset: 10pt,

  [*Fase del pipeline*], [*ID de la fase*],

  [Preprocesamiento], [preprocessing],
  [Rebalanceo], [rebalancing],
  [Mutación de etiquetas], [mutval],
  [Entrenamiento], [training],
  [Evaluación de los resultados], [evaluation]
)
