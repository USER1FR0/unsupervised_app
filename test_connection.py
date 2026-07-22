from src.db.connection import ping, get_db

if ping():
    db = get_db()
    print(f"OK - Conectado a MongoDB Atlas")
    print(f"Base de datos: {db.name}")
    print(f"Colecciones existentes: {db.list_collection_names()}")
else:
    print("ERROR - No se pudo conectar a MongoDB Atlas")