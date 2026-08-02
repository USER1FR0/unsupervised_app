from datetime import datetime, timezone
from bson import ObjectId
from src.db.connection import get_db

COLLECTION_NAME = "models"


def get_collection():
    return get_db()[COLLECTION_NAME]


def save_metadata(doc: dict) -> str:
    """Guarda el metadata del modelo. Retorna el _id insertado."""
    if "trained_at" not in doc:
        doc["trained_at"] = datetime.now(timezone.utc)
    result = get_collection().insert_one(doc)
    return str(result.inserted_id)


def find_all() -> list:
    """Lista todos los modelos ordenados del más reciente al más antiguo."""
    return list(get_collection().find().sort("trained_at", -1))


def find_by_id(model_id: str) -> dict:
    """Busca un modelo por su _id."""
    return get_collection().find_one({"_id": ObjectId(model_id)})


def delete_by_id(model_id: str) -> bool:
    """Elimina un modelo del historial."""
    result = get_collection().delete_one({"_id": ObjectId(model_id)})
    return result.deleted_count > 0


def count() -> int:
    return get_collection().count_documents({})


def ensure_indexes():
    coll = get_collection()
    coll.create_index("trained_at")
    coll.create_index("algorithm")