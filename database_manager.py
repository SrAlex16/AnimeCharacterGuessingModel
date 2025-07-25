import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import os
import pandas as pd # Para manejar DataFrames si es necesario en las funciones de BD

# --- Configuración de rutas (CONSTANTES) ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_PATH = os.path.join(BASE_PATH, 'firebase_credentials.json')

# --- Inicialización de Firebase ---
# Variable global para la instancia de Firestore
db = None

def initialize_firestore():
    """
    Inicializa la conexión a Firestore usando las credenciales de Firebase.
    """
    global db
    if firebase_admin._apps:  # Verifica si Firebase ya ha sido inicializado
        print("Firebase ya inicializado.")
        db = firestore.client()
        return db

    if not os.path.exists(CREDENTIALS_PATH):
        print(f"Error: Archivo de credenciales no encontrado en {CREDENTIALS_PATH}")
        return None
    
    try:
        cred = credentials.Certificate(CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase inicializado correctamente.")
        return db
    except Exception as e:
        print(f"Error al inicializar Firebase: {e}")
        return None
    
# --- Funciones de manejo de datos ---
def save_character_data(character_data: pd.DataFrame):
    """
    Guarda los datos de los personajes en una colección de Firestore.
    Cada fila del DataFrame se convierte en un documento.
    La colección será 'anime_characters_data'.
    """
    
    if db is None:
        print("Firestore no está inicializado. Por favor, llama a initialize_firestore() primero.")
        return False
    
    collection_ref = db.collection('anime_characters_data')
    batch = db.batch()  # Usamos un batch para mejorar la eficiencia

    print(f"Guardando {len(character_data)} personajes en Firestore...")
    for index, row in character_data.iterrows():
        # Usamos el ID para facilitar la búsqueda y actualización
        doc_id = str(row['mal_id'])

        # Preparamos los datos a guardar
        # Convertimos a dict y manejamos NaN si es necesario
        # Solo guardamos las columnas relevantes
        data_to_save = {
            'mal_id': row['mal_id'],
            'image_jpg_url': row['image_jpg_url'],
            'about': row['about'],
            'gender': row['gender'], # 'chico', 'chica', 'unclassified'
            'gender_numeric': int(row['gender_numeric']) if pd.notna(row['gender_numeric']) else None, # Convertir a int
            'local_image_path': row['local_image_path'] if pd.notna(row['local_image_path']) else None,
            'status': 'initial_load' # Un campo para rastrear el estado
        }

        #Elimnar claves con valores None si no queremos guardarlas
        data_to_save = {k: v for k, v in data_to_save.items() if v is not None}

        # Creamos una referencia al documento
        doc_ref = collection_ref.document(doc_id)

        # Añadimos el documento al batch
        batch.set(doc_ref, data_to_save)

        if (index + 1) % 500 == 0:  # Cada 100 documentos, hacemos commit
            batch.commit()
            batch = db.batch()
            print(f"Guardados {index + 1} personajes hasta ahora...")

    # Hacemos commit del último batch
    try:
        batch.commit()
        print(f"Todos los {len(character_data)} personajes guardados correctamente en Firestore.")
        return True
    except Exception as e:
        print(f"Error al guardar personajes en Firestore: {e}")
        return False
    
def load_all_character_data():
    """
    Carga todos los datos de personajes desde Firestore.
    Devuelve un DataFrame de pandas con los datos.
    """
    
    if db is None:
        print("Firestore no está inicializado. Por favor, llama a initialize_firestore() primero.")
        return None
    
    print("Cargando datos de personajes desde Firestore...")
    # Obtenemos una referencia a la colección
    collection_ref = db.collection('anime_characters_data')
    docs = collection_ref.stream() # Obtenemos todos los documentos de la colección

    data = []
    for doc in docs:
        data.append(doc.to_dict())

    if data:
        df = pd.DataFrame(data)
        print(f"Cargados {len(df)} personajes desde Firestore.")
        return df
    else:
        print("No se encontraron datos de personajes en Firestore.")
        return pd.DataFrame()  # Retorna un DataFrame vacío si no hay datos
    
def update_character_status(mal_id: int, new_status: str, gender_prediction = None):
    """
    Actualiza el estado de un personaje específico en Firestore.
    Puede incluir una predicción de género si es relevante.
    """

    if db is None:
        print("Firestore no está inicializado. Por favor, llama a initialize_firestore() primero.")
        return False

    doc_ref = db.collection('anime_characters_data').document(str(mal_id))  # Convertimos mal_id a string para Firestore
    # Preparamos los datos a actualizar
    update_data = {'status': new_status}  

    if gender_prediction is not None:
        update_data['gender_prediction'] = gender_prediction # guardamos las predicciones de la IA

    try:
        doc_ref.update(update_data)
        print(f"Estado del personaje {mal_id} actualizado a '{new_status}'.")
        return True
    except Exception as e:
        print(f"Error al actualizar el estado del personaje {mal_id}: {e}")
        return False
    
# Bloque de ejecución principal para probar el módulo directamente
if __name__ == "__main__":
    # Inicializa Firestore
    firestore_db = initialize_firestore()
    if firestore_db:
        print("\nPrueba de carga de datos (si ya existen)...")
        df_loaded = load_all_character_data()
        print(df_loaded.head())

        test_data = pd.DataFrame([
            {'mal_id': 999999, 'image_jpg_url': 'test_url.jpg', 'about': 'Test character male', 'gender': 'chico', 'gender_numeric': 0, 'local_image_path': '/path/to/test.jpg', 'status': 'test_entry'},
            {'mal_id': 999998, 'image_jpg_url': 'test_url2.jpg', 'about': 'Test character female', 'gender': 'chica', 'gender_numeric': 1, 'local_image_path': '/path/to/test2.jpg', 'status': 'test_entry'}
        ])
        save_character_data(test_data)
        update_character_status(999999, 'processed_by_test')