from datetime import datetime, timezone
from bson import ObjectId
from src.db.connection import get_db

COLLECTION_NAME = "models"


def get_collection():
    return get_db()[COLLECTION_NAME]


def save_metadata(doc: dict) -> str:
    """Guarda el metadata del modelo. Requiere model_name y dataset_source.
    Retorna el _id insertado."""
    if not doc.get("model_name"):
        raise ValueError("model_name es obligatorio.")
    if not doc.get("dataset_source"):
        raise ValueError("dataset_source es obligatorio.")

    if "trained_at" not in doc:
        doc["trained_at"] = datetime.now(timezone.utc)

    result = get_collection().insert_one(doc)
    return str(result.inserted_id)


def find_all() -> list:
    """Lista todos los modelos, del mas reciente al mas antiguo."""
    return list(get_collection().find().sort("trained_at", -1))


def find_by_id(model_id: str) -> dict:
    return get_collection().find_one({"_id": ObjectId(model_id)})


def find_by_name(model_name: str) -> dict:
    return get_collection().find_one({"model_name": model_name})


def name_exists(model_name: str) -> bool:
    return get_collection().count_documents({"model_name": model_name}) > 0


def delete_by_id(model_id: str) -> bool:
    result = get_collection().delete_one({"_id": ObjectId(model_id)})
    return result.deleted_count > 0


def count() -> int:
    return get_collection().count_documents({})


def ensure_indexes():
    coll = get_collection()
    coll.create_index("trained_at")
    coll.create_index("algorithm")
    coll.create_index("dataset_source")
    coll.create_index("model_name", unique=True, sparse=True)


def readable_model_name(m: dict) -> str:
    """Construye un nombre corto y descriptivo para mostrar en UI.

    Prioriza el model_name explicito. Si no existe, arma uno con
    algoritmo, clusters, registros y fecha corta.
    """
    if m.get("model_name"):
        return m["model_name"]

    algo = (m.get("algorithm") or "modelo").lower()
    metrics = m.get("metrics") or {}
    k = metrics.get("n_clusters", "?")
    n = m.get("n_records", "?")
    ts = m.get("trained_at")
    fecha = ts.strftime("%d/%m %H:%M") if ts else ""
    parts = [algo, f"k={k}", f"n={n}"]
    if fecha:
        parts.append(fecha)
    return " · ".join(str(p) for p in parts)


def readable_dataset_name(m: dict) -> str:
    """Devuelve el nombre del dataset origen limpio (sin extension, sin '?')."""
    ds = m.get("dataset_source") or ""
    if not ds or ds == "?":
        return "sin dataset"
    return ds.replace(".csv", "")
