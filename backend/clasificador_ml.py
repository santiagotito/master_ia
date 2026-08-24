"""Ensamblado de texto y clasificación SVM de Nivel 1 (Entrega 2, ratificada).

Ratifica clean_text() de PROYECTO/Clasificador_Piezas_ML.ipynb (celda 10),
el patrón de texto para piezas reales de 2026 (celda 37: version+anunciante,
sin 'item' -- ese campo solo existe en el Excel histórico de entrenamiento,
ingesta.listar_piezas() no lo produce), y el entrenamiento/predicción de los
4 pipelines TF-IDF+SVM (celdas 12, 21, 35) con los hiperparámetros óptimos
ya validados por GridSearchCV en la Entrega 2 -- no se vuelve a buscar la
grilla acá.

AD-1 (aditivo): este módulo entrena pipelines NUEVOS en el repo de código;
no modifica ni reentrena nada dentro del notebook original.
"""
import re

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import CONF_THRESHOLD_NIVEL1, EXCEL_PATH, MIN_SOPORTE_ENTRENAMIENTO, SEED, TARGET_COLS


def clean_text(text):
    """Limpia y normaliza texto: mayúsculas, sin tildes, sin caracteres
    especiales. Ratificado tal cual de la celda 10 del notebook."""
    if text is None:
        return ""
    t = str(text).upper().strip()
    for src, dst in [('Á', 'A'), ('É', 'E'), ('Í', 'I'), ('Ó', 'O'), ('Ú', 'U'), ('Ü', 'U'), ('Ñ', 'N')]:
        t = t.replace(src, dst)
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def ensamblar_texto(pieza, enriquecimiento=None):
    """Combina el texto base de una pieza (version + anunciante, patrón de
    la celda 37) con el texto enriquecido por IA de Epic 1 (AC #1).

    `pieza`: dict de `ingesta.listar_piezas()` -- usa 'version'/'anunciante'.
    `enriquecimiento`: dict de `describir_imagen()` (texto_ocr,
    descripcion_visual) y/o `transcribir_audio()` (transcripcion_audio), o
    None si la pieza todavía no fue procesada por Epic 1. Campos ausentes o
    en None se omiten sin insertar la palabra "None" (AC #4).
    """
    partes = [clean_text(pieza.get('version')), clean_text(pieza.get('anunciante'))]

    if enriquecimiento:
        for campo in ('texto_ocr', 'descripcion_visual', 'transcripcion_audio'):
            valor = enriquecimiento.get(campo)
            if valor:
                partes.append(clean_text(valor))

    return ' '.join(p for p in partes if p).strip()


def make_tfidf():
    """TfidfVectorizer con los hiperparámetros óptimos ya validados en la
    Entrega 2 (GridSearchCV, ver tabla "Hiperparámetros Óptimos" de la
    Entrega 1): ngram_range=(1,2), max_features=5000. Ratificado, no se
    vuelve a correr la búsqueda de grilla."""
    return TfidfVectorizer(
        ngram_range=(1, 2), max_features=5000,
        sublinear_tf=True, strip_accents='unicode', min_df=2,
    )


def _preparar_df_clean(excel_path=None):
    """Carga y limpia el dataset histórico, ratificado de las celdas 6 y 10
    del notebook (rename de columnas por posición, texto combinado
    versiones+anunciante+item, filtrado de filas completas). Aislada de
    `_cargar_dataset()` para que la Story 3.1 pueda reutilizar el MISMO
    dataset limpio sin duplicar esta lógica (Task 1, Story 3.1)."""
    excel_path = excel_path or EXCEL_PATH
    df_raw = pd.read_excel(excel_path, sheet_name='SUBBASE', engine='openpyxl')

    col_map = {
        df_raw.columns[0]: 'sector', df_raw.columns[1]: 'categoria',
        df_raw.columns[2]: 'anio', df_raw.columns[3]: 'mes',
        df_raw.columns[4]: 'anunciante', df_raw.columns[5]: 'marca',
        df_raw.columns[6]: 'item', df_raw.columns[7]: 'versiones',
        df_raw.columns[8]: 'llave', df_raw.columns[9]: 'tipo_campania',
        df_raw.columns[10]: 'modalidad', df_raw.columns[11]: 'grado',
        df_raw.columns[12]: 'carrera', df_raw.columns[13]: 'detalle1',
        df_raw.columns[14]: 'detalle2', df_raw.columns[15]: 'carrera_general',
    }
    df = df_raw.rename(columns=col_map).copy()

    df['texto'] = (
        df['versiones'].apply(clean_text) + " " +
        df['anunciante'].apply(clean_text) + " " +
        df['item'].apply(clean_text)
    ).str.strip()

    required = list(TARGET_COLS.values()) + ['texto']
    df_clean = df.dropna(subset=required).copy()
    df_clean = df_clean[df_clean['texto'] != ''].reset_index(drop=True)

    for c in list(TARGET_COLS.values()) + ['texto']:
        df_clean[c] = df_clean[c].astype(object)

    return df_clean


def _split_train_test_idx(df_clean):
    """MISMO split (test_size=0.2, random_state=SEED, stratify=tipo_campania)
    que ya generó los resultados documentados en la Entrega 2. Aislado para
    que `_cargar_dataset()` y `_metadata_test()` (Story 3.1) usen exactamente
    los mismos índices de train/test."""
    idx = np.arange(len(df_clean))
    return train_test_split(
        idx, test_size=0.2, random_state=SEED,
        stratify=df_clean["tipo_campania"].tolist(),
    )


def _cargar_dataset(excel_path=None):
    """Carga y prepara el dataset histórico, ratificado de las celdas 6, 10
    y 12 del notebook. Devuelve (X_train, X_test, y_trains, y_tests)."""
    df_clean = _preparar_df_clean(excel_path)
    idx_train, idx_test = _split_train_test_idx(df_clean)

    X_all = np.array(df_clean["texto"].tolist())
    X_train, X_test = X_all[idx_train], X_all[idx_test]
    y_trains = {col: np.array(df_clean[col].tolist())[idx_train] for col in TARGET_COLS.values()}
    y_tests = {col: np.array(df_clean[col].tolist())[idx_test] for col in TARGET_COLS.values()}

    return X_train, X_test, y_trains, y_tests


def _metadata_test(excel_path=None):
    """Devuelve los metadatos identificadores (`anunciante`, `versiones`,
    `mes`, `anio`) + etiquetas reales de las filas del split de TEST, en el
    mismo orden que `y_tests` de `_cargar_dataset()` -- usado por la Story
    3.1 (`piloto.py`) para emparejar el conjunto de test con archivos reales
    de Drive. Aplica el MISMO split que `_cargar_dataset()` (misma semilla),
    así que los índices de test coinciden exactamente."""
    df_clean = _preparar_df_clean(excel_path)
    _idx_train, idx_test = _split_train_test_idx(df_clean)

    columnas = ['anunciante', 'versiones', 'mes', 'anio'] + list(TARGET_COLS.values())
    return df_clean.iloc[idx_test][columnas].reset_index(drop=True)


def entrenar_pipelines(excel_path=None):
    """Entrena los 4 pipelines TF-IDF+SVM (uno por columna J/K/L/M) sobre el
    dataset histórico, ratificado de la celda 21. Devuelve
    (pipelines, soporte_clases): `pipelines` es {cn: Pipeline entrenado},
    `soporte_clases` es {col: {clase: cantidad_ejemplos_train}}, usado por
    `clasificar_nivel1()` para decidir si hay soporte suficiente."""
    X_train, X_test, y_trains, y_tests = _cargar_dataset(excel_path)

    pipelines = {}
    soporte_clases = {}
    for cn, col in TARGET_COLS.items():
        pipe = Pipeline([
            ('tfidf', make_tfidf()),
            ('clf', CalibratedClassifierCV(
                LinearSVC(C=5, max_iter=3000, class_weight='balanced', random_state=SEED),
                cv=None,
            )),
        ])
        pipe.fit(X_train, y_trains[col])
        pipelines[cn] = pipe

        clases, cuentas = np.unique(y_trains[col], return_counts=True)
        soporte_clases[col] = dict(zip(clases.tolist(), cuentas.tolist()))

    return pipelines, soporte_clases, X_test, y_tests


def clasificar_nivel1(texto, pipelines, soporte_clases):
    """Clasifica un texto ya ensamblado (`ensamblar_texto()`, Story 2.1) con
    los 4 pipelines SVM, ratificado de `predecir_pieza()` (celda 35).

    Devuelve {cn: {'pred': str, 'conf': float, 'es_final': bool}}. `es_final`
    es True solo si la confianza supera `CONF_THRESHOLD_NIVEL1` Y la clase
    predicha tuvo al menos `MIN_SOPORTE_ENTRENAMIENTO` ejemplos en el set de
    entrenamiento (AC #2) -- ambos umbrales son placeholders a calibrar en
    el piloto de Epic 3, ver config.py.
    """
    resultados = {}
    for cn, col in TARGET_COLS.items():
        pipe = pipelines[cn]
        pred = pipe.predict([texto])[0]
        conf = float(pipe.predict_proba([texto])[0].max())
        soporte = soporte_clases.get(col, {}).get(pred, 0)

        es_final = conf >= CONF_THRESHOLD_NIVEL1 and soporte >= MIN_SOPORTE_ENTRENAMIENTO
        resultados[cn] = {'pred': pred, 'conf': conf, 'es_final': es_final}

    return resultados
