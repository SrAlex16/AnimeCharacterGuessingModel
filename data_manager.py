import pandas as pd
import os
import re
import requests
import numpy as np # Necesario para np.nan
# Importamos el módulo database_manager
import database_manager 

# --- Configuración de rutas y directorios (CONSTANTES) ---
# Se define aquí para que sea accesible dentro de este módulo
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_PATH, 'anime_characters.csv')
# Usamos 'img' como nombre de la carpeta de caché para ser consistente con el proyecto del usuario
IMAGE_CACHE_DIR = os.path.join(BASE_PATH, 'img') 

# Creamos el directorio de imágenes si no existe
os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)

# --- Función para extraer el género (UNIFICADA) ---
def extract_gender_from_about(text):
    """
    Extrae el género de un personaje en base a la columna 'about' del dataset.
    Devuelve 'chico', 'chica' o 'unclassified' si no se encuentra información clara.
    """
    if not isinstance(text, str):
        return 'unclassified' # Si no es texto (ej. NaN), lo consideramos 'unclassified'
    
    text_lower = text.lower()

    male_keywords = ['he', 'him', 'his', 'boy', 'male', 'man', 'masculine','prince','king','brother','father']
    female_keywords = ['she', 'her', 'hers', 'girl', 'female', 'feminine', 'woman', 'princess', 'queen', 'sister', 'mother']

    male_score = sum(text_lower.count(keyword) for keyword in male_keywords)
    female_score = sum(text_lower.count(keyword) for keyword in female_keywords)

    if male_score > female_score and male_score > 0:
        return 'chico'
    elif female_score > male_score and female_score > 0:
        return 'chica'
    else:
        return 'unclassified'

# --- Función para descargar una imagen (MEJORADA para evitar descargas redundantes) ---
def download_image(url, character_id, image_type='jpg'):
    """
    Descarga una imagen desde una URL y la guarda localmente.
    Retorna la ruta del archivo guardado si tiene éxito, None en caso contrario.
    Si el archivo ya existe, no lo descarga de nuevo.
    """
    filename = os.path.join(IMAGE_CACHE_DIR, f"{character_id}.{image_type}") # Usamos IMAGE_CACHE_DIR

    if os.path.exists(filename):
        return filename
    
    if not url or pd.isna(url):
        return None
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()

        with open(filename, 'wb') as out_file: # Usamos out_file para consistencia
            for chunk in response.iter_content(chunk_size=8192):
                out_file.write(chunk)
        return filename
    except requests.exceptions.RequestException as e: # Usamos requests.exceptions
        # print(f"Error al descargar la imagen de {character_id}: {e}") # Descomentar para depurar errores de descarga
        return None
    except Exception as e:
        # print(f"Error inesperado al guardar la imagen {character_id}: {e}") # Descomentar para depurar otros errores
        return None

# --- Función principal para preparar los datos ---
def prepare_data():
    """
    Carga el dataset (desde SQLite o CSV), extrae géneros, descarga imágenes
    y prepara los DataFrames para entrenamiento e inferencia posterior.
    Retorna df_for_training_cleaned y df_for_later_inference.
    """
    # 1. Inicializar la base de datos SQLite
    db_conn = database_manager.initialize_database()
    if db_conn is None:
        print("No se pudo inicializar la base de datos SQLite. Saliendo de la preparación de datos.")
        return None, None

    # 2. Intentar cargar datos desde SQLite
    csv_df = database_manager.load_all_character_data() 
    
    # Flag para saber si los datos se cargaron por primera vez desde CSV
    first_load_from_csv = False 

    if csv_df.empty:
        print("\nNo se encontraron datos en SQLite. Cargando desde CSV local por primera vez...")
        try:
            csv_df = pd.read_csv(CSV_PATH) 
            print("Dataset cargado desde CSV correctamente. Primeras filas:")
            print(csv_df.head())
            first_load_from_csv = True
        except FileNotFoundError:
            print(f"Error: El archivo {CSV_PATH} no se encontró.")
            print("Asegúrate de que el archivo CSV esté en el mismo directorio que el script.")
            return None, None # Retornar None si hay error
    else:
        print("\nDatos cargados desde SQLite. Primeras filas:")
        print(csv_df.head())
    
    # --- Aplicar la función de extracción de género al DataFrame ---
    # Solo aplica si la columna 'gender' no existe o si queremos recalcularla
    if 'gender' not in csv_df.columns:
        print("\nExtrayendo etiquetas de género de la columna 'about'...")
        csv_df['gender'] = csv_df['about'].apply(extract_gender_from_about)
    else:
        print("\nColumna 'gender' ya existe. Saltando extracción de género.")

    # --- Convertir etiquetas de género a numéricas (necesario para guardar en SQLite) ---
    gender_mapping = {'chico': 0, 'chica': 1}
    # Aseguramos que 'gender_numeric' exista para todos, incluso 'unclassified' (será None)
    csv_df['gender_numeric'] = csv_df['gender'].map(gender_mapping)

    # --- Añadir columna local_image_path si no existe (para el primer guardado) ---
    if 'local_image_path' not in csv_df.columns:
        csv_df['local_image_path'] = None # O np.nan para consistencia
    
    # --- Añadir columna status si no existe (para el primer guardado) ---
    if 'status' not in csv_df.columns:
        csv_df['status'] = 'initial_load' # Estado inicial para todos los cargados

    # --- PASO CRÍTICO: Guardar/actualizar todos los datos iniciales en SQLite si es la primera carga desde CSV ---
    # Esto asegura que los documentos existan antes de intentar actualizarlos individualmente
    if first_load_from_csv:
        print("\nGuardando datos iniciales del CSV en SQLite...")
        database_manager.save_character_data(csv_df) # Usamos csv_df aquí
        print("Datos iniciales guardados en SQLite.")
        # No es necesario recargar desde SQLite inmediatamente, ya que `save_character_data`
        # ya actualiza el DataFrame en la base de datos. `csv_df` ya contiene los datos.


    # --- Dividir el DataFrame en clasificados y no clasificados (después del guardado inicial) ---
    df_for_training = csv_df[csv_df['gender'].isin(['chico', 'chica'])].copy()
    df_for_later_inference = csv_df[csv_df['gender'] == 'unclassified'].copy()

    # Mostramos estadísticas del dataset
    print("\nEstadísticas del dataset:")
    print(f"Total de personajes en el dataset: {len(csv_df)}") # Usamos csv_df aquí
    print(f"Total de personajes masculinos (para entrenamiento): {len(df_for_training[df_for_training['gender'] == 'chico'])}")
    print(f"Total de personajes femeninos (para entrenamiento): {len(df_for_training[df_for_training['gender'] == 'chica'])}")
    print(f"Total de personajes no clasificados (para inferencia posterior): {len(df_for_later_inference)}")

    # --- Bucle para descargar imágenes ---
    print("\nIniciando la descarga de imágenes para el conjunto de entrenamiento...")
    downloaded_image_paths = []
    download_errors = 0
    total_images_to_download = len(df_for_training)

    # NUEVA LÓGICA: Cargar estados y rutas locales de SQLite una sola vez
    print("Cargando estados de descarga de imágenes desde SQLite para optimizar el bucle...")
    # Creamos un diccionario para un acceso rápido por mal_id
    # Si la base de datos es muy grande, esto podría consumir mucha RAM,
    # pero para 200k registros es manejable.
    training_data_from_db = {}
    if not csv_df.empty:
        # Filtramos solo los que nos interesan para el entrenamiento
        temp_df = csv_df[csv_df['mal_id'].isin(df_for_training['mal_id'])]
        for _, row_db in temp_df.iterrows():
            training_data_from_db[str(row_db['mal_id'])] = {
                'status': row_db.get('status'),
                'local_image_path': row_db.get('local_image_path')
            }
    print(f"Estados de {len(training_data_from_db)} personajes cargados de SQLite.")


    for i, (original_index, row) in enumerate(df_for_training.iterrows()):
        character_id = row['mal_id']
        image_url = row['image_jpg_url']
        
        # Usar la información cargada en memoria
        mal_id_str = str(character_id)
        db_info = training_data_from_db.get(mal_id_str, {})
        current_status = db_info.get('status')
        local_path_from_db = db_info.get('local_image_path')

        if current_status == 'image_downloaded':
            if local_path_from_db and os.path.exists(local_path_from_db):
                downloaded_image_paths.append(local_path_from_db)
                df_for_training.loc[original_index, 'local_image_path'] = local_path_from_db
                if (i + 1) % 100 == 0:
                    print(f"Progreso: {i + 1}/{total_images_to_download} imágenes procesadas (saltadas). Errores: {download_errors}")
                continue # Saltar al siguiente personaje
        
        # Si no está marcado como descargado o el archivo no existe localmente, intentar descargar
        local_path = download_image(image_url, character_id, 'jpg')
        if local_path:
            downloaded_image_paths.append(local_path)
            df_for_training.loc[original_index, 'local_image_path'] = local_path
            df_for_training.loc[original_index, 'status'] = 'image_downloaded'
        else:
            download_errors += 1
            df_for_training.loc[original_index, 'status'] = 'image_download_failed'
        
        # Imprimir progreso cada 100 imágenes usando el contador 'i'
        if (i + 1) % 100 == 0:
            print(f"Progreso: {i + 1}/{total_images_to_download} imágenes procesadas. Errores: {download_errors}")
        
    print(f"\nDescarga de imágenes completada. Total descargadas: {len(downloaded_image_paths)}. Errores: {download_errors}")
    print(f"Las imágenes se están guardando en: {IMAGE_CACHE_DIR}")

    # Filtramos las filas donde la imagen no se pudo descargar
    df_for_training_cleaned = df_for_training.dropna(subset=['local_image_path']).copy()
    print(f"\nTotal de imágenes válidas para entrenamiento después de la descarga: {len(df_for_training_cleaned)}")

    # --- Convertir etiquetas de género a numéricas (para el DataFrame final de entrenamiento) ---
    gender_mapping = {'chico': 0, 'chica': 1}
    df_for_training_cleaned['gender_numeric'] = df_for_training_cleaned['gender'].map(gender_mapping)


    print("\nDistribución de géneros numéricos en el conjunto de entrenamiento limpio:")
    print(df_for_training_cleaned['gender_numeric'].value_counts())

    print("\nPrimeras filas del conjunto de entrenamiento limpio con etiquetas numéricas:")
    print(df_for_training_cleaned[['gender', 'gender_numeric', 'local_image_path']].head())

    # 3. Guardar/actualizar todos los datos procesados en SQLite (incluyendo los 'unclassified')
    # Aseguramos que 'gender_numeric' y 'local_image_path' tengan valores consistentes para el concat
    if 'local_image_path' not in df_for_later_inference.columns:
        df_for_later_inference['local_image_path'] = None 
    if 'gender_numeric' not in df_for_later_inference.columns:
        df_for_later_inference['gender_numeric'] = None 
    if 'status' not in df_for_later_inference.columns: # Asegurar que 'status' también esté
        df_for_later_inference['status'] = 'unclassified_by_text' # O un estado apropiado

    all_final_data = pd.concat([df_for_training_cleaned, df_for_later_inference])
    print("\nGuardando/actualizando todos los datos procesados en SQLite (final)...")
    database_manager.save_character_data(all_final_data)

    print("\nPreparación de datos completada en data_manager.py.")
    
    return df_for_training_cleaned, df_for_later_inference

# Esto asegura que prepare_data() solo se ejecute si el script se corre directamente
# No se ejecutará si se importa desde otro archivo
if __name__ == "__main__":
    df_training, df_inference = prepare_data()
    if df_training is not None:
        print("\nDataFrames preparados y listos para ser usados por model_trainer.py.")