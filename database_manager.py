import sqlite3
import pandas as pd
import os
import numpy as np # Importar numpy para np.nan

# --- Configuración de rutas (CONSTANTES) ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# Nombre del archivo de la base de datos SQLite
DB_FILE_NAME = 'anime_characters.db' 
DB_PATH = os.path.join(BASE_PATH, DB_FILE_NAME)

# --- Inicialización de la base de datos SQLite ---
conn = None # Variable global para la conexión a la base de datos

def initialize_database():
    """
    Inicializa la conexión a la base de datos SQLite y crea la tabla si no existe.
    Esta función debe llamarse una vez al inicio de tu aplicación.
    """
    global conn
    if conn is not None:
        print("La base de datos SQLite ya está inicializada.")
        return conn

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Creamos la tabla si no existe
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS characters (
                mal_id INTEGER PRIMARY KEY,
                image_jpg_url TEXT,
                about TEXT,
                gender TEXT,
                gender_numeric INTEGER,
                local_image_path TEXT,
                status TEXT
            )
        ''')
        conn.commit()
        print(f"Base de datos SQLite inicializada correctamente en: {DB_PATH}")
        return conn
    except Exception as e:
        print(f"Error al inicializar la base de datos SQLite: {e}")
        return None

# --- Funciones para interactuar con SQLite ---

def save_character_data(character_data: pd.DataFrame):
    """
    Guarda o actualiza los datos de los personajes en la base de datos SQLite.
    Utiliza INSERT OR REPLACE para actualizar si el mal_id ya existe.
    """
    if conn is None:
        print("Error: Base de datos SQLite no inicializada. Llama a initialize_database() primero.")
        return False

    print(f"\nGuardando/Actualizando {len(character_data)} personajes en SQLite...")
    
    # Convertir el DataFrame a una lista de tuplas para inserción
    # Aseguramos el orden de las columnas para que coincida con la tabla
    columns = [
        'mal_id', 'image_jpg_url', 'about', 'gender', 
        'gender_numeric', 'local_image_path', 'status'
    ]
    # Rellenar NaN con None para que SQLite los maneje correctamente
    data_to_save = character_data[columns].replace({np.nan: None}).values.tolist()

    cursor = conn.cursor()
    try:
        # Usamos INSERT OR REPLACE para que si el mal_id ya existe, se actualice la fila
        # y si no existe, se inserte una nueva.
        # CORRECCIÓN: Eliminado el 'f' innecesario al inicio de la cadena SQL
        cursor.executemany('''
            INSERT OR REPLACE INTO characters (
                mal_id, image_jpg_url, about, gender, gender_numeric, local_image_path, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', data_to_save)
        conn.commit()
        print(f"Guardado/Actualización de datos completado. Total de documentos guardados/actualizados: {len(character_data)}")
        return True
    except Exception as e:
        print(f"Error al guardar datos en SQLite: {e}")
        return False

def load_all_character_data():
    """
    Carga todos los documentos de la tabla 'characters' de SQLite
    y los devuelve como un DataFrame de Pandas.
    """
    if conn is None:
        print("Error: Base de datos SQLite no inicializada. Llama a initialize_database() primero.")
        return pd.DataFrame()

    print(f"\nCargando todos los personajes desde SQLite (Tabla: characters)...")
    try:
        df = pd.read_sql_query("SELECT * FROM characters", conn)
        print(f"Cargados {len(df)} personajes desde SQLite.")
        return df
    except pd.io.sql.DatabaseError as e:
        print(f"Error al cargar datos desde SQLite: {e}")
        print("Esto podría deberse a que la tabla aún no existe o está vacía.")
        return pd.DataFrame() 
    except Exception as e:
        print(f"Error inesperado al cargar datos desde SQLite: {e}")
        return pd.DataFrame()

def update_character_status(mal_id: int, new_status: str, gender_prediction=None, local_image_path=None):
    """
    Actualiza el estado de un personaje específico en SQLite.
    Puede incluir una predicción de género y la ruta local de la imagen.
    """
    if conn is None:
        print("Error: Base de datos SQLite no inicializada. Llama a initialize_database() primero.")
        return False
    
    cursor = conn.cursor()
    update_query = "UPDATE characters SET status = ?"
    params = [new_status]
    
    if gender_prediction is not None:
        update_query += ", gender_prediction = ?"
        params.append(gender_prediction)
    if local_image_path is not None:
        update_query += ", local_image_path = ?"
        params.append(local_image_path)
    
    update_query += " WHERE mal_id = ?"
    params.append(mal_id)

    try:
        cursor.execute(update_query, params)
        conn.commit()
        # print(f"Estado del personaje {mal_id} actualizado a '{new_status}'.") # Descomentar para ver actualizaciones individuales
        return True
    except Exception as e:
        print(f"Error al actualizar el personaje {mal_id} en SQLite: {e}") 
        return False

# Bloque de ejecución principal para probar el módulo directamente
if __name__ == "__main__":
    db_conn = initialize_database()
    if db_conn:
        print("\nPrueba de carga de datos (si ya existen)...")
        df_loaded = load_all_character_data()
        print(df_loaded.head())

        print("\nGuardando datos de prueba...")
        test_data = pd.DataFrame([
            {'mal_id': 999999, 'image_jpg_url': 'test_url.jpg', 'about': 'Test character male', 'gender': 'chico', 'gender_numeric': 0, 'local_image_path': '/path/to/test.jpg', 'status': 'test_entry'},
            {'mal_id': 999998, 'image_jpg_url': 'test_url2.jpg', 'about': 'Test character female', 'gender': 'chica', 'gender_numeric': 1, 'local_image_path': '/path/to/test2.jpg', 'status': 'test_entry'}
        ])
        save_character_data(test_data)
        
        print("\nActualizando estado de prueba...")
        update_character_status(999999, 'processed_by_test_sqlite')
        
        print("\nCargando datos de prueba actualizados...")
        df_updated = load_all_character_data()
        print(df_updated[df_updated['mal_id'].isin([999999, 999998])])
