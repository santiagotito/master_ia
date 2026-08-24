"""Configuración centralizada del clasificador híbrido ML+IA de RAM.

Ratifica las constantes ya usadas en PROYECTO/Clasificador_Piezas_ML.ipynb
(celda 4) — no reinventa rutas ni columnas objetivo.
"""
import os

try:
    from google.colab import drive  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    BASE_DIR = '/content/drive/MyDrive/1. RAM'
else:
    # Ruta real confirmada del Drive de RAM montado localmente (Google Drive
    # Desktop) -- corregido en la Story 2.2, la ruta anterior era un
    # placeholder nunca validado y no existía.
    BASE_DIR = r"G:\Mi unidad\1. RAM"

EXCEL_PATH = os.path.join(BASE_DIR, "Subbase UIDE Oct 2025.xlsm")
FOLDER_2026 = os.path.join(BASE_DIR, "2026", "2026")
OUTPUT_PATH = os.path.join(BASE_DIR, "Predicciones_2026_ML.xlsx")

# Excel con el pipeline híbrido completo (Nivel 1 + Nivel 2, Story 3.4),
# generado en Colab -- la app local (sin GPU) lo lee tal cual en vez de
# reejecutar el VLM/LLM, que no corre razonablemente sin GPU.
HIBRIDO_PATH = os.path.join(BASE_DIR, "Predicciones_2026_ML_hibrido.xlsx")

# Carpeta de caché de pesos de modelo (AD-3 / NFR3 de ARCHITECTURE-SPINE.md):
# persiste entre sesiones de Colab para no re-descargar varios GB cada vez.
HF_HOME_DRIVE = os.path.join(BASE_DIR, "modelos_cache")

TARGET_COLS = {'J': 'tipo_campania', 'K': 'modalidad', 'L': 'grado', 'M': 'carrera'}
TARGET_LABELS = {
    'J': 'J — Tipo de Campaña', 'K': 'K — Modalidad',
    'L': 'L — Grado',           'M': 'M — Carrera',
}

# Mapa de extensión -> tipo de pieza, ratificado de la celda 37 del notebook
# y ampliado tras validar contra datos reales de RAM (Story 1.1): se
# detectaron 20 piezas reales (.gif, .wmv) que el mapa original no cubría
# y quedaban excluidas en silencio.
# Toda ampliación de tipos de archivo debe editarse SOLO aquí (AD-2).
EXT_MAP = {
    '.mp3': 'Audio', '.mp4': 'Video', '.wmv': 'Video', '.gif': 'Video',
    '.jpg': 'Imagen', '.jpeg': 'Imagen',
    '.png': 'Imagen', '.pdf': 'PDF',
}

# Carpetas de mes dentro de FOLDER_2026, ratificado de la celda 37 del notebook.
MESES_MAP = {
    'ENE': 'enero',   'FEB': 'febrero',  'MAR': 'marzo',
    'ABR': 'abril',   'MAY': 'mayo',     'JUN': 'junio',
    'JUL': 'julio',   'AGO': 'agosto',   'SEP': 'septiembre',
    'OCT': 'octubre', 'NOV': 'noviembre', 'DIC': 'diciembre',
}

SEED = 42

# Umbral de confianza de OCR/ASR para decidir revisión humana (Epic 2/3).
# Valor placeholder: se calibra empíricamente en el piloto de la Story 3.1-3.2,
# no es un valor definitivo.
CONF_THRESHOLD_OCR_ASR = 0.5

# Umbrales de clasificación de Nivel 1 (SVM).
# CALIBRADO CON EVIDENCIA REAL (Story 3.1, piloto n=50 sobre piezas reales
# con etiqueta conocida): con el placeholder original (0.85), el SVM
# resolvía muy pocos casos y mandaba el resto a Nivel 2 (LLM), que rindió
# peor que el propio SVM en esos mismos casos (ver hallazgo en
# docs/implementation-artifacts/3-1-piloto-muestra-test.md). Un barrido de
# umbrales sobre la misma muestra mostró que 0.6 resuelve 49-50/50 piezas
# en Nivel 1 con 96-98% de exactitud en las 4 columnas -- muy por encima
# de dejarlas en manos del Nivel 2 actual. Sigue siendo un valor a
# recalibrar si la muestra crece (Story 3.4) o si el Nivel 2 mejora.
CONF_THRESHOLD_NIVEL1 = 0.6  # confianza mínima para aceptar la predicción del SVM como final
MIN_SOPORTE_ENTRENAMIENTO = 5  # ejemplos mínimos de la clase predicha en el set de entrenamiento
