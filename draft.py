# Cargar el modelo entrenado
best_model_path = 'ruta_al_mejor_modelo.h5'
best_model = tf.keras.models.load_model(best_model_path)

# Cargar los datos de prueba
X_test = ...  # Cargar tus características de prueba
y_test = ...  # Cargar tus etiquetas de prueba

# Hacer predicciones
predictions = best_model.predict(X_test)
predicted_labels = np.round(predictions).flatten()  # Convertir las probabilidades en etiquetas binarias

# Calcular métricas de evaluación
accuracy = accuracy_score(y_test, predicted_labels)
conf_matrix = confusion_matrix(y_test, predicted_labels)

# Mostrar los resultados
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)