import pandas as pd
import os
import requests
import database_manager

# Ruta relativa segura basada en la ubicación del script
base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, 'anime_characters.csv')
csv = pd.read_csv(csv_path) # Cargo el dataset
IMAGE_CACHE_DIR = os.path.join(base_path, 'img') # Ruta de las imágenes

# Creamos el directorio de imágenes si no existe
if not os.path.exists(IMAGE_CACHE_DIR):
    os.makedirs(IMAGE_CACHE_DIR)

def extract_gender_from_about(text):
    """
    Extrae el género de un personaje en base a la columna 'about' del dataset.
    Devuelve 'chico', 'chica' o 'chicx' si no se encuentra información clara.
    """
    if not isinstance(text, str):
        return 'unclassified' # Si no es texto, lo consideramos desconocido
    
    text = text.lower() # Convertimos a minúsculas para facilitar la búsqueda

    # Buscamos palabras clave para determinar el género
    male_keywords = ['he', 'him', 'his', 'boy', 'male', 'man', 'masculine','prince','king','brother','father']
    female_keywords = ['she', 'her', 'hers', 'girl', 'female', 'feminine', 'woman', 'princess', 'queen', 'sister', 'mother']

    if any(keyword in text for keyword in male_keywords):
        return 'chico' # Chico
    elif any(keyword in text for keyword in female_keywords):
        return 'chica' # Chica
    else:
        return 'unclassified' # No se pudo clasificar por el texto about

# Aplicamos la función a la columna 'about' del DataFrame
csv['gender'] = csv['about'].apply(extract_gender_from_about)

# Dividimos el DataFrame en tres DataFrames según si están clasificados o no
df_for_training = csv[csv['gender'].isin(['chico', 'chica'])].copy()
df_for_later_inference = csv[csv['gender'] == 'unclassified'].copy()

# Mostramos estadísticas del dataset
print("Estadísticas del dataset:")
print(f"Total de personajes en el dataset: {len(csv)}")
print(f"Total de personajes masculinos: {len(df_for_training[df_for_training['gender'] == 'chico'])}")
print(f"Total de personajes femeninos: {len(df_for_training[df_for_training['gender'] == 'chica'])}")
print(f"Total de personajes no clasificados: {len(df_for_later_inference)}")

# Descargamos las imágenes de los personajes
def download_image(url, character_id, image_type='jpg'):
    """
    Descarga una imagen desde una URL y la guarda localmente.
    Retorna la ruta del archivo guardado si tiene éxito, None en caso contrario.
    Si el archivo ya existe, no lo descarga de nuevo.
    """
    filename = os.path.join(IMAGE_CACHE_DIR, f"{character_id}.{image_type}") # Ruta completa del archivo a guardar
    
    if os.path.exists(filename):
        print(f"Imagen {character_id}.{image_type} ya existe, saltando descarga.") # Descomentar para ver mensajes de salto
        return filename # Si ya existe, simplemente devuelve la ruta
    
    if not url or pd.isna(url): # Añadimos pd.isna para manejar valores NaN en la URL
        return None
    
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status() # Verificamos si la solicitud fue exitosa
        
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        return filename
    except requests.exceptions.RequestException as e: # Usamos requests.exceptions
        print(f"Error al descargar la imagen de {character_id}: {e}") # Descomentar para depurar errores de descarga
        return None
    except Exception as e:
        print(f"Error inesperado al guardar la imagen {character_id}: {e}") # Descomentar para depurar otros errores
        return None

def prepare_data():
    """
    Prepara los datos para el entrenamiento del modelo.
    """
    # Carga del dataset
    try:
        csv = pd.read_csv(csv_path)
        print("Dataset cargado correctamente. Primeras filas:")
        print(csv.head())
    except FileNotFoundError:
        print(f"Error: El archivo {csv_path} no se encontró.")
        return None
    
    # Aplicamos la función de extracción de género
    print("Extrayendo género de los personajes...")
    csv['gender'] = csv['about'].apply(extract_gender_from_about)

    # Dividimos el dataframe en clasificados y no clasificados
    df_for_training = csv[csv['gender'].isin(['chico', 'chica'])].copy()
    df_for_later_inference = csv[csv['gender'] == 'unclassified'].copy()

    # Mostramos estadísticas del dataset
    print("Estadísticas del dataset:")
    print(f"Total de personajes en el dataset: {len(csv)}")
    print(f"Total de personajes masculinos (para entrenamiento): {len(df_for_training[df_for_training['gender'] == 'chico'])}")
    print(f"Total de personajes femeninos (para entrenamiento): {len(df_for_training[df_for_training['gender'] == 'chica'])}")
    print(f"Total de personajes no clasificados (para inferencia posterior): {len(df_for_later_inference)}")

    # Bucle para descargar imágenes
    print("Descargando imágenes de personajes...")
    downloaded_image_paths = []
    download_errors = 0
    total_images_to_download = len(df_for_training) # Guardamos la longitud una vez

    # Iteramos sobre los personajes
    for i, (original_index, row) in enumerate(df_for_training.iterrows()):
        character_id = row['mal_id'] # ID del personaje
        image_url = row['image_jpg_url'] # URL de la imagen del personaje
        
        # Descargamos la imagen
        image_path = download_image(image_url, character_id, image_type='jpg')
        
        # Si la imagen se descargó correctamente, la añadimos a la lista
        if image_path:
            downloaded_image_paths.append(image_path)
        else:
            download_errors += 1
        
        # Mostramos el resultado de la descarga cada 100 imágenes
        if (i + 1) % 100 == 0:
            print(f"Progreso: {i + 1}/{len(df_for_training)} imágenes procesadas. Errores: {download_errors}")
        
    print(f"\nDescarga de imágenes completada. Total descargadas: {len(downloaded_image_paths)}. Errores: {download_errors}")
    print(f"Las imágenes se están guardando en: {IMAGE_CACHE_DIR}")

    # Guardamos las rutas de las imágenes descargadas y sus etiquetas para no tener que descargarlas de nuevo si el script se detiene
    df_for_training['local_image_path'] = df_for_training.apply(
        lambda row: os.path.join(IMAGE_CACHE_DIR, f"{row['mal_id']}.jpg") if os.path.exists(os.path.join(
            IMAGE_CACHE_DIR, f"{row['mal_id']}.jpg")) else None, axis=1)

    # Filtramos las filas donde la imagen no se pudo descargar
    df_for_training_cleaned = df_for_training.dropna(subset=['local_image_path']).copy()
    print(f"\nTotal de imágenes válidas para entrenamiento después de la descarga: {len(df_for_training_cleaned)}")

    # Convertimos las etiquetas de género a valores numéricos
    gender_mapping = {'chico': 0, 'chica': 1}
    df_for_training_cleaned['gender_numeric'] = df_for_training_cleaned['gender'].map(gender_mapping)
    print("Etiquetas de género convertidas a valores numéricos.")
    print(df_for_training_cleaned['gender_numeric'].value_counts())
    print("\nPrimeras filas del conjunto de entrenamiento limpio con etiquetas numéricas:")
    print(df_for_training_cleaned[['gender', 'gender_numeric', 'local_image_path']].head())
    print("\nPreparación de datos completada en data_manager.py.")

    # Guardar / actualizar los datos en Firestore
    print("Guardando datos de personajes en Firestore...")

    # Combinamos los dataframes para guardar todo en una sola colección y actualizar sus estados
    # Es importante que df_for_training_cleaned y df_for_later_inference mantengan las mismas columnas
    # para que save_character_data pueda procesarlos.
    # Aseguramos que 'gender_numeric' exista en df_for_later_inference para la unión
    if 'gender_numeric' not in df_for_later_inference.columns:
        df_for_later_inference['gender_numeric'] = None # O np.nan
    if 'local_image_path' not in df_for_later_inference.columns:
        df_for_later_inference['local_image_path'] = None # O np.nan

    all_processed_data = pd.concat([df_for_training_cleaned, df_for_later_inference])
    database_manager.save_character_data(all_processed_data)
    print("Datos guardados en Firestore.")
    
    return df_for_training_cleaned, df_for_later_inference

# Esto asegura que prepare_data() solo se ejecute si el script se corre directamente
# No se ejecutará si se importa desde otro archivo
if __name__ == "__main__":
    df_training, df_inference = prepare_data()
    if df_training is not None:
        print("\nDataFrames preparados y listos para ser usados por model_trainer.py.")