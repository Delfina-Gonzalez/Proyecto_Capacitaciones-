from pathlib import Path

# 📌 Raíz del proyecto (siempre case6382)
PROJECT_ROOT = Path(__file__).parent.resolve()

# 📂 Datos crudos (Excel originales)
DATA_DIR = PROJECT_ROOT / "data"

# 📂 Datos procesados (parquet / csv)
PROCESSED_DIR = PROJECT_ROOT / "data_processed"

# Crear carpetas si no existen
DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

