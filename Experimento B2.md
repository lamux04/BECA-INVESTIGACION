# PROCEDIMIENTO
1. Preparación de los datos (eliminar filas y columnas).
2. Undersampling de forma aleatoria. Con esto terminaríamos con 50% benigno y 50% maligno.
3. Oversampling usando SMOTE. Aquí haríamos oversampling de las clases malignas para que la suma de todas ellas sean equitativas a la clase benigna.

# MODELO
Voy a usar el modelo LLama en lugar del modelo T5. La principal diferencia es que el modelo T5, dado una cadena de texto devuelve una respuesta. Sin embargo, el modelo LLama, dada una cadena de texto intenta continuar la cadena.