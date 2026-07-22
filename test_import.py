from src.sheets_importer import fetch_sheet_dataframe, dataframe_to_documents
from src.db.response_repository import upsert_many, ensure_indexes, count

print("Descargando datos de Google Sheets...")
df = fetch_sheet_dataframe()
print(f"Filas descargadas: {len(df)}")
print(f"Columnas: {list(df.columns)}")

if len(df) == 0:
    print("El Sheet está vacío.")
else:
    print("\nPrimera fila:")
    print(df.iloc[0].to_dict())

    print("\nCreando índices...")
    ensure_indexes()

    print("\nInsertando en MongoDB...")
    docs = dataframe_to_documents(df)
    result = upsert_many(docs)
    print(f"Resultado: {result}")

    print(f"\nTotal de documentos en MongoDB: {count()}")