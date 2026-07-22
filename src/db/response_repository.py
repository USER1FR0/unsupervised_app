from datetime import datetime
from typing import Iterable
from pymongo import UpdateOne
from src.db.connection import get_db

COLLECTION_NAME = "responses"


def get_collection():
    return get_db()[COLLECTION_NAME]


def upsert_many(documents: Iterable[dict]) -> dict:
    """Inserta o actualiza documentos usando timestamp como clave única."""
    coll = get_collection()
    operations = []
    for doc in documents:
        operations.append(UpdateOne(
            {"submitted_at": doc["submitted_at"]},
            {"$set": doc},
            upsert=True,
        ))

    if not operations:
        return {"inserted": 0, "updated": 0, "total": 0}

    result = coll.bulk_write(operations, ordered=False)
    return {
        "inserted": result.upserted_count,
        "updated": result.modified_count,
        "total": len(operations),
    }


def count() -> int:
    return get_collection().count_documents({})


def find_all() -> list:
    return list(get_collection().find().sort("submitted_at", -1))


def ensure_indexes():
    coll = get_collection()
    coll.create_index("submitted_at", unique=True)
    coll.create_index("arquetipo")
    coll.create_index("estado")