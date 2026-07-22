import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[1] / "config" / ".env"
load_dotenv(ENV_PATH)

SHEET_ID = os.getenv("SHEET_ID")

RAW_ANSWER_COLUMNS = [f"q{i}" for i in range(1, 21)]
DIMENSION_COLUMNS = ["O", "C", "E", "A", "N"]
META_COLUMNS = ["arquetipo", "edad", "genero", "estado", "municipio"]


def _csv_url() -> str:
    if not SHEET_ID:
        raise RuntimeError("SHEET_ID no está definido en config/.env")
    return f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid=0"


def fetch_sheet_dataframe() -> pd.DataFrame:
    url = _csv_url()
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    response.encoding = "utf-8"

    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    return df


def _row_to_document(row: pd.Series) -> dict:
    raw_answers = [int(row[c]) for c in RAW_ANSWER_COLUMNS]
    scores = {dim: float(row[dim]) for dim in DIMENSION_COLUMNS}
    submitted_at = pd.to_datetime(row["timestamp"], dayfirst=True).to_pydatetime()

    return {
        "submitted_at": submitted_at,
        "raw_answers": raw_answers,
        **scores,
        "arquetipo": str(row["arquetipo"]),
        "edad": int(row["edad"]),
        "genero": str(row["genero"]),
        "estado": str(row["estado"]),
        "municipio": str(row["municipio"]),
        "source": "google_form",
        "synthetic": False,
    }


def dataframe_to_documents(df: pd.DataFrame) -> list:
    return [_row_to_document(row) for _, row in df.iterrows()]