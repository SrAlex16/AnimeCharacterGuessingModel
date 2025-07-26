import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import os
import data_manager # Módulo para manejar la base de datos SQLite
import monitor # Módulo que monitorea el rendimiento del modelo

# --- Configuración del modelo y procesamiento de imágenes ---
IMAGE_SIZE = (128, 128)  # Tamaño de las imágenes de entrada
BATCH_SIZE = 32  # Tamaño del lote para entrenamiento
EPOCHS = 10  # Número de épocas para el entrenamiento
NUM_CLASSES = 2  # Número de clases (masculino, femenino)

# --- Funciones de preparación de datos para TensorFlow ---
def preprocess_image_for_tf(image_path, label):
    """
    Preprocesa una imagen para TensorFlow.
    Carga una imagen, la decodifica, redimensiona y normaliza.
    Se usa en el pipeline de tf.data
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3) # Decodifica la imagen JPEG con 3 canales (RGB)
    img = tf.image.resize(img, IMAGE_SIZE) # Redimensiona la imagen al tamaño especificado
    img = tf.cast(img, tf.float32) / 255.0  # Normalización a [0, 1]
    return img

def create_tf_datasets(image_paths, labels):
    """
    Crea datasets de TensorFlow para entrenamiento, validación y pruebas.
    """
    # Dividir el dataset en entreamiento, validación y prueba
    # Primero dividimos en entrenamiento (80%) y validación (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Luego dividimos el conjunto temporal en validación (10%) y prueba (10%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\n--- División del Dataset para TensorFlow ---")
    print(f"Imágenes de entrenamiento: {len(X_train)}")
    print(f"Imágenes de validación: {len(X_val)}")
    print(f"Imágenes de prueba: {len(X_test)}")

    # Crear datasets de TensorFlow
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    #Mapear la función de preprocesamiento a cada dataset
    train_dataset = train_dataset.map(preprocess_image_for_tf, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_dataset = val_dataset.map(preprocess_image_for_tf, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_dataset = test_dataset.map(preprocess_image_for_tf, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print("\nDatasets de TensorFlow creados:")
    print(f" - Entrenamiento: {len(train_dataset)}")
    print(f" - Validación: {len(val_dataset)}")
    print(f" - Prueba: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset

# --- Construcción del modelo de Red Neuronal Convolucional (CNN) ---
def build_cnn_model(input_shape, num_classes):
    """
    Construye un modelo de Red Neuronal Convolucional (CNN) para clasificación de imágenes.
    """
    model = models.Sequential([
        # Primera capa convolucional y de pooling
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Segunda capa convolucional y de pooling
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Tercera capa convolucional y de pooling
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Aplanar la salida para las capas densas
        layers.Flatten(),
        
        # Capas densas (fully connected)
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Regularización para evitar overfitting
        layers.Dense(num_classes, activation='softmax') # Capa de salida para clasificación binaria (0 o 1)
    ])
    return model

# --- Entrenamiento del modelo ---
def train_model():
    """
    Orquesta la preparación de datos, construcción, entrenamiento y evaluación del modelo.
    """
    print("\n--- Iniciando el entrenamiento del modelo ---")
    # Cargar los datos desde SQLite
    df_training_cleaned, df_for_later_inference = data_manager.prepare_data()

    if df_training_cleaned is None or df_training_cleaned.empty:
        print("Error: No hay datos suficientes para entrenar el modelo.")
        return
    
    # Extraer rutas de imágenes y etiquetas
    image_paths = df_training_cleaned['local_image_path'].values
    labels = df_training_cleaned['gender_numeric'].values

    # Crear datasets de TensorFlow
    train_dataset, val_dataset, test_dataset = create_tf_datasets(image_paths, labels)

    # Construir el modelo CNN
    input_shape = IMAGE_SIZE + (3,)  # Tamaño de la imagen + número de canales (RGB)
    model = build_cnn_model(input_shape, NUM_CLASSES)
    model.summary()  # Muestra la arquitectura del modelo

    # Compilar el modelo
    print("\nCompilando el modelo...")
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # Usamos sparse_categorical_crossentropy para etiquetas enteras
                  metrics=['accuracy'])
    print("Modelo compilado.")

    resource_monitor = monitor.ResourceMonitorCallback(log_interval_batches=100)

    # Entrenar el modelo
    print("\nIniciando el entrenamiento del modelo...")
    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, verbose=1, callbacks=[resource_monitor]) # Aquí se añade el callback
    print("Entrenamiento completado.")

    # Evaluar el modelo en el conjunto de prueba
    print("\nEvaluando el modelo en el conjunto de prueba...")
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
    print(f"Pérdida en el conjunto de prueba: {test_loss:.4f}")
    print(f"Precisión en el conjunto de prueba: {test_accuracy:.4f}")

    # Guardar el modelo entrenado
    model_save_path = os.path.join(data_manager.BASE_PATH, 'anime_gender_classifier_model.h5')
    model.save(model_save_path)
    print(f"Modelo guardado como {model_save_path}")
    print("Proceso de entrenamiento y evaluación completado")

    # Graficar los datos
    plot_dir = os.path.join(data_manager.BASE_PATH, 'plots')
    os.makedirs(plot_dir, exist_ok=True) 

    monitor.plot_training_history(
        history, 
        save_path=os.path.join(plot_dir, 'training_history.png')
    )
    monitor.plot_resource_usage(
        resource_monitor, 
        save_path=os.path.join(plot_dir, 'resource_usage.png')
    )

    return history

# Bloque de ejecución principal para probar el módulo directamente
if __name__ == "__main__":
    train_model()