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

= Información de los modelos

En este documento irá información sobre todos los modelos del proyecto.

#table(
  columns: 7,
  align: (center, center, center, center, center, center, center),
  inset: 10pt,

  [*Nombre modelo*], [*Dataset*], [*Tokens*], [*Batch*], [*Fold*], [*Accuracy*], [*Notas*]
)