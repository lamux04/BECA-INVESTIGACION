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

== Dataset CIC-IDS-2017 (eda_v1.ipynb)

=== Información general

#image("/assets/image.png")


=== Distribución de las clases

#image("/assets/image-1.png")

=== Problemas típicos

- Homogeneizar nombres de columnas.
- Eliminar columnas inútiles: Las columnas que se indican en el paper de Leopoldo ya han sido eliminadas.
- Eliminar columnas con valores únicos.
- Eliminar columnas con alta correlación.
- Eliminar filas con valores nulos.
- Eliminar columnas con valores infinito.

=== Columnas

| Destination Port
| Flow Duration
| Total Fwd Packets
| Total Backward Packets
| Total Length of Fwd Packets
| Total Length of Bwd Packets
| Fwd Packet Length Max
| Fwd Packet Length Min
| Fwd Packet Length Mean
| Fwd Packet Length Std
| Bwd Packet Length Max
| Bwd Packet Length Min
| Bwd Packet Length Mean
| Bwd Packet Length Std
| Flow Bytes/s
| Flow Packets/s
| Flow IAT Mean
| Flow IAT Std
| Flow IAT Max
| Flow IAT Min
| Fwd IAT Total
| Fwd IAT Mean
| Fwd IAT Std
| Fwd IAT Max
| Fwd IAT Min
| Bwd IAT Total
| Bwd IAT Mean
| Bwd IAT Std
| Bwd IAT Max
| Bwd IAT Min
| Fwd PSH Flags
| Bwd PSH Flags
| Fwd URG Flags
| Bwd URG Flags
| Fwd Header Length
| Bwd Header Length
| Fwd Packets/s
| Bwd Packets/s
| Min Packet Length
| Max Packet Length
| Packet Length Mean
| Packet Length Std
| Packet Length Variance
| FIN Flag Count
| SYN Flag Count
| RST Flag Count
| PSH Flag Count
| ACK Flag Count
| URG Flag Count
| CWE Flag Count
| ECE Flag Count
| Down/Up Ratio
| Average Packet Size
| Avg Fwd Segment Size
| Avg Bwd Segment Size
| Fwd Header Length.1
| Fwd Avg Bytes/Bulk
| Fwd Avg Packets/Bulk
| Fwd Avg Bulk Rate
| Bwd Avg Bytes/Bulk
| Bwd Avg Packets/Bulk
| Bwd Avg Bulk Rate
| Subflow Fwd Packets
| Subflow Fwd Bytes
| Subflow Bwd Packets
| Subflow Bwd Bytes
| Init_Win_bytes_forward
| Init_Win_bytes_backward
| act_data_pkt_fwd
| min_seg_size_forward
| Active Mean
| Active Std
| Active Max
| Active Min
| Idle Mean
| Idle Std
| Idle Max
| Idle Min
| Label |

== Dataset CIC-IDS-2018 (eda_v2.ipynb)

