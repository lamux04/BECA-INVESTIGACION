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

= Datasets de ciberseguridad

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