"""Carga y limpieza del dataset."""

import os
import re
import pandas as pd
import numpy as np
from src.config import TARGET_COLS, EXCEL_PATH, SEED


def clean_text(text):
    """Limpia y normaliza texto: mayúsculas, sin tildes, sin caracteres especiales."""
    if pd.isna(text):
        return ""
    t = str(text).upper().strip()
    for src, dst in [('Á', 'A'), ('É', 'E'), ('Í', 'I'), ('Ó', 'O'), ('Ú', 'U'), ('Ü', 'U'), ('Ñ', 'N')]:
        t = t.replace(src, dst)
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def parse_filename(filename):
    """Extrae anunciante y versión del nombre de un archivo publicitario."""
    base = os.path.splitext(os.path.basename(str(filename)))[0]
    if ' - ' in base:
        anunciante, resto = base.split(' - ', 1)
        version = re.sub(r'\([^)]*\)\s*$', '', resto).strip()
        version = re.sub(r'[-_]\s*(FB|CIG|IG|TW|YT|\d+X\d+)\s*$', '', version).strip()
    else:
        anunciante, version = '', base
    return anunciante.strip(), version.strip()


def load_and_preprocess_data(excel_path=EXCEL_PATH):
    """Carga y procesa la hoja SUBBASE de Excel."""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {excel_path}")

    df_raw = pd.read_excel(excel_path, sheet_name='SUBBASE', engine='openpyxl')

    # Mapeo exacto por posición de columnas
    col_map = {
        df_raw.columns[0]:  'sector',
        df_raw.columns[1]:  'categoria',
        df_raw.columns[2]:  'anio',
        df_raw.columns[3]:  'mes',
        df_raw.columns[4]:  'anunciante',
        df_raw.columns[5]:  'marca',
        df_raw.columns[6]:  'item',
        df_raw.columns[7]:  'versiones',
        df_raw.columns[8]:  'llave',
        df_raw.columns[9]:  'tipo_campania',
        df_raw.columns[10]: 'modalidad',
        df_raw.columns[11]: 'grado',
        df_raw.columns[12]: 'carrera',
        df_raw.columns[13]: 'detalle1',
        df_raw.columns[14]: 'detalle2',
        df_raw.columns[15]: 'carrera_general',
    }
    df = df_raw.rename(columns=col_map).copy()

    # Construcción de texto combinado para el modelo TF-IDF
    df['texto'] = (
        df['versiones'].apply(clean_text) + " " +
        df['anunciante'].apply(clean_text) + " " +
        df['item'].apply(clean_text)
    ).str.strip()

    # Filtrar registros con etiquetas requeridas y texto no vacío
    required = list(TARGET_COLS.values()) + ['texto']
    df_clean = df.dropna(subset=required).copy()
    df_clean = df_clean[df_clean['texto'] != ''].reset_index(drop=True)

    # ── FIX CRÍTICO: forzar dtype=object (numpy puro) en todas las columnas
    #    de texto. Pandas 3.x usa PyArrow por defecto y sklearn no lo tolera.
    for c in list(TARGET_COLS.values()) + ['texto']:
        df_clean[c] = df_clean[c].astype(object)

    # Métricas y muestras por consola
    print(f"Filas con datos completos: {len(df_clean):,} / {len(df):,} ({len(df_clean)/len(df)*100:.1f}%)")
    print(f"Longitud promedio del texto: {df_clean['texto'].str.split().str.len().mean():.1f} tokens")
    print(f"Dtype columna texto: {df_clean['texto'].dtype}")

    print("\n=== EJEMPLOS DE TEXTO DE ENTRADA ===")
    sample_size = min(5, len(df_clean))
    for _, row in df_clean.sample(sample_size, random_state=SEED).iterrows():
        print(f"  Texto : {row['texto'][:70]}")
        print(f"  J={row['tipo_campania']} | K={row['modalidad']} | L={row['grado']} | M={str(row['carrera'])[:35]}")
        print()

    return df_clean