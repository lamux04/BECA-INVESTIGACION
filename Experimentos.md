| Exp | Dataset | Undersampling | Oversampling | Benigno (%) | Maligno (%) | Modelo | Métrica | Valor | Observaciones                        |
| --- | ------- | ------------- | ------------ | ----------- | ----------- | ------ | ------- | ----- | ------------------------------------ |
| A   | 40M     | Random        | SMOTE igual  | 6.66        | 93.33       | t5     |         | —     | Demasiado costoso computacionalmente |
| B   | 5M      | Random        | SMOTE 50/50  | 50          | 50          | t5     |         | —     | —                                    |

# EXPERIMENTO A (40M)
1. Preparación (eliminar filas y columnas).
2. Undersampling random.
3. Oversampling SMOTE (todo igual al benigno).

# EXPERIMENTO B (5M)
1. Preparación.
2. Undersampling random.
3. Oversampling SMOTE (benigno 50% y malignos repartidos por igual).

# UNDERSAMPLING
- Random.
- Tonken links.
- Edifed nerarest neighbour.
