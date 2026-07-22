import os
from pathlib import Path
from urllib.parse import quote_plus
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / "config" / ".env"
load_dotenv(ENV_PATH)

MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")
MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_DB = os.getenv("MONGO_DB", "unsupervised_app")

_client = None


def _build_uri() -> str:
    if not all([MONGO_USER, MONGO_PASS, MONGO_HOST]):
        raise RuntimeError("Faltan variables MONGO_USER, MONGO_PASS o MONGO_HOST en config/.env")
    user = quote_plus(MONGO_USER)
    password = quote_plus(MONGO_PASS)
    return f"mongodb+srv://{user}:{password}@{MONGO_HOST}/?retryWrites=true&w=majority&appName=Cluster0"


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(_build_uri(), serverSelectionTimeoutMS=5000)
    return _client


def get_db():
    return get_client()[MONGO_DB]


def ping() -> bool:
    try:
        get_client().admin.command("ping")
        return True
    except ConnectionFailure:
        return False