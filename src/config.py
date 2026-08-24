""" Mantiene los mapeos y rutas configurables para los archivos del proyecto
"""
import os
from pathlib import Path

# Directorios principales
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
GRAFICOS_DIR = BASE_DIR / "graficos"

# Directorio donde se guardan/leen las piezas multimedia (imágenes, vídeos, PDFs, audios).
# Preferimos el Drive real de RAM (dataset completo) si está montado en esta
# máquina; si no, caemos a la copia local reducida de `data/2026` (útil para
# correr la demo en una máquina sin el Drive de RAM montado).
_DRIVE_2026 = Path(r"G:\Mi unidad\1. RAM\2026\2026")
PIEZAS_DIR = _DRIVE_2026 if _DRIVE_2026.is_dir() else DATA_DIR / "2026"

# Crear carpetas si no existen
MODELS_DIR.mkdir(parents=True, exist_ok=True)
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)
PIEZAS_DIR.mkdir(parents=True, exist_ok=True)

# Rutas de archivos
EXCEL_PATH = DATA_DIR / "Subbase UIDE Oct 2025.xlsm"
MODEL_PATH = MODELS_DIR / "model_pipelines.pkl"
OUTPUT_EXCEL_PATH = DATA_DIR / "Predicciones_2026_ML.xlsx"

# Mapeos
TARGET_COLS = {'J': 'tipo_campania', 'K': 'modalidad', 'L': 'grado', 'M': 'carrera'}
TARGET_LABELS = {
    'J': 'J — Tipo de Campaña', 'K': 'K — Modalidad',
    'L': 'L — Grado',           'M': 'M — Carrera',
}
SEED = 42

MESES_MAP = {
    'ENE': 'enero',  'FEB': 'febrero', 'MAR': 'marzo',
    'ABR': 'abril',  'MAY': 'mayo',    'JUN': 'junio',
    'JUL': 'julio',  'AGO': 'agosto',  'SEP': 'septiembre',
    'OCT': 'octubre','NOV': 'noviembre','DIC': 'diciembre',
}

EXT_MAP = {
    '.mp3': 'Audio', '.mp4': 'Video',
    '.jpg': 'Imagen', '.jpeg': 'Imagen',
    '.png': 'Imagen', '.pdf': 'PDF'
}
