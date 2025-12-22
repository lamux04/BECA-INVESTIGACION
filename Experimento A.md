# PROCEDIMIENTO
1. Preparación de los datos (eliminar filas y columnas).
2. Undersampling de forma aleatoria. Con esto terminaríamos con 50% benigno y 50% maligno.
3. Oversampling usando SMOTE. Aquí haríamos oversampling de las clases malignas para que sean equitativas a la clase benigna.

# PARTICIÓN DE DATOS
A la hora de particionar los datos tenemos 2 niveles de partición:
- 20% de test.
- 80% de entrenamiento.
	- 5 K folds para hacer cross-validation.

Para hacer las particiones se ha usado la estrategia stratified para que sean equitativamente repartidos por todos los subconjuntos.

# RESULTADOS
Después de intentar hacer el entrenamiento durante 30 días el proceso no ha terminado. Por tanto, dejamos este experimento para otro momento.