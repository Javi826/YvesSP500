# Utiliza la imagen oficial de TensorFlow Serving
FROM tensorflow/serving

# Copia tu modelo a la ubicación esperada por TensorFlow Serving
COPY tf_serving /models/my_model

# Exponer el puerto 8501 para que TensorFlow Serving sea accesible desde fuera del contenedor
EXPOSE 8501

# Ejecutar TensorFlow Serving cuando se inicie el contenedor
CMD ["tensorflow_model_server", "--rest_api_port=8501", "--model_name=my_model", "--model_base_path=/models/my_model"]
