import os
import re
import sys

import pandas as pd

from src.config import (
    TARGET_COLS,
    TARGET_LABELS,
    MODEL_PATH,
    EXT_MAP,
    MESES_MAP,
    PIEZAS_DIR,
    OUTPUT_EXCEL_PATH,
)
from src.data_loader import clean_text

# Backend real del sistema híbrido (Nivel 1 SVM calibrado + ingesta con
# recorrido recursivo de subcarpetas) -- integrado desde repo-clasificador-ram
# en vez de mantener una segunda copia de la lógica de clasificación acá.
# Se agrega backend/ al sys.path (no un paquete con imports relativos)
# porque los módulos del backend usan imports planos (`from config import
# ...`), igual que en el notebook de Colab.
_BACKEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import clasificador_ml as backend_ml  # noqa: E402
import config as backend_config  # noqa: E402
import ingesta as backend_ingesta  # noqa: E402


def load_trained_models(model_path=MODEL_PATH):
    """Entrena los 4 pipelines TF-IDF+SVM sobre el histórico real (backend
    validado, Story 2.2/3.1-3.2) -- reemplaza el .pkl estático anterior, que
    no aplicaba umbral de confianza ni soporte mínimo por clase. Entrenar
    tarda ~10-15s en CPU y `app.py` ya lo cachea con `@st.cache_resource`,
    así que solo corre una vez por sesión de Streamlit.

    `model_path` se mantiene en la firma por compatibilidad, ya no se usa.
    """
    pipelines, soporte_clases, _X_test, _y_tests = backend_ml.entrenar_pipelines()
    return {"pipelines": pipelines, "soporte_clases": soporte_clases}


def parse_filename(filename):
    """Extrae anunciante y versión del nombre de archivo.
    Formato esperado: ANUNCIANTE - VERSIÓN(FORMATO).ext
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    if ' - ' in base:
        anunciante, resto = base.split(' - ', 1)
        # Quitar (FORMATO) al final
        version = re.sub(r'\([^)]*\)\s*$', '', resto).strip()
        # Quitar sufijos de plataforma: -FB, -CIG, -IG, -300X250, etc.
        version = re.sub(r'[-_]\s*(FB|CIG|IG|TW|YT|\d+X\d+)\s*$', '', version).strip()
    else:
        anunciante, version = '', base
    return anunciante.strip(), version.strip()


def predecir_pieza_single(models, version, anunciante='', item=''):
    """Predice J, K, L, M para una sola pieza de publicidad (retorna dict
    estructurado). Usa el mismo umbral calibrado (Story 3.1: 0.6) que el
    resto del sistema para marcar `es_final` -- antes esta función no
    aplicaba ningún umbral, aceptaba siempre la predicción cruda del SVM."""
    pipelines, soporte_clases = models["pipelines"], models["soporte_clases"]
    texto = (backend_ml.clean_text(version) + ' ' + backend_ml.clean_text(anunciante) + ' ' + backend_ml.clean_text(item)).strip()
    nivel1 = backend_ml.clasificar_nivel1(texto, pipelines, soporte_clases)

    resultados = {}
    for cn, col in TARGET_COLS.items():
        pipe = pipelines[cn]
        proba = pipe.predict_proba([texto])[0]
        top3 = sorted(zip(pipe.classes_, proba), key=lambda x: -x[1])[:3]
        r = nivel1[cn]
        resultados[cn] = {
            'label': TARGET_LABELS[cn],
            'pred': r['pred'],
            'conf': r['conf'],
            'top3': top3,
            'es_final': r['es_final'],
        }

    return resultados


def predecir_pieza(version, anunciante='', item='', models=None, verbose=True):
    """
    Predice J, K, L, M para una pieza y muestra el output formateado en consola.
    """
    if models is None:
        models = load_trained_models()

    resultados = predecir_pieza_single(models, version=version, anunciante=anunciante, item=item)

    if verbose:
        print(f"{'─'*58}")
        print(f"  PIEZA      : {version[:50]}")
        print(f"  ANUNCIANTE : {anunciante[:50]}")
        print(f"{'─'*58}")
        for cn in TARGET_COLS:
            r = resultados[cn]
            bar = '█' * int(r['conf'] * 20)
            print(f"  {r['label']}")
            print(f"    → {r['pred']:<35} [{r['conf']*100:5.1f}%] {bar}  {'(Nivel 1 final)' if r['es_final'] else '(necesitaría Nivel 2)'}")
            for cls, p in r['top3'][1:]:
                print(f"       {cls:<35} ({p*100:.1f}%)")
        print()

    return resultados


def predecir_lote_dataframe(models, df_input):
    """Genera predicciones sobre un DataFrame con columnas 'version' y
    'anunciante', usando el mismo umbral calibrado que el resto del sistema
    (Nivel 1 únicamente -- sin GPU local no hay Nivel 2 en vivo)."""
    pipelines, soporte_clases = models["pipelines"], models["soporte_clases"]
    df_res = df_input.copy()

    textos = (
        df_res['version'].apply(backend_ml.clean_text) + ' ' +
        df_res['anunciante'].apply(backend_ml.clean_text)
    ).str.strip().tolist()

    resultados_por_columna = {cn: [] for cn in TARGET_COLS}
    for texto in textos:
        nivel1 = backend_ml.clasificar_nivel1(texto, pipelines, soporte_clases)
        for cn in TARGET_COLS:
            resultados_por_columna[cn].append(nivel1[cn])

    for cn, col in TARGET_COLS.items():
        df_res[col] = [r['pred'] for r in resultados_por_columna[cn]]
        df_res[f'conf_{cn}'] = [round(r['conf'] * 100, 1) for r in resultados_por_columna[cn]]
        df_res[f'nivel_{cn}'] = [1 if r['es_final'] else None for r in resultados_por_columna[cn]]

    return df_res


def _clasificar_piezas(models, piezas):
    """Clasifica una lista de piezas (dicts con al menos 'archivo', 'tipo',
    'anunciante', 'version') con Nivel 1 (SVM calibrado). Compartida por
    `procesar_carpeta_piezas()` (piezas del servidor) y
    `clasificar_archivos_subidos()` (piezas subidas por el usuario) para no
    duplicar la lógica de clasificación en dos lugares."""
    pipelines, soporte_clases = models["pipelines"], models["soporte_clases"]
    records = []

    for p in piezas:
        texto = backend_ml.ensamblar_texto(p, enriquecimiento=None)
        nivel1 = backend_ml.clasificar_nivel1(texto, pipelines, soporte_clases)

        fmt_match = re.search(r'\(([^)]+)\)', p['archivo'])
        row = {
            'anio': 2026,
            'mes': MESES_MAP.get(p.get('mes', ''), p.get('mes', '')),
            'anunciante': p.get('anunciante', ''),
            'version': p.get('version', ''),
            'tipo_archivo': p['tipo'],
            'formato': fmt_match.group(1) if fmt_match else '',
            'archivo': p['archivo'],
            'ruta_local': p.get('ruta', ''),
        }
        for cn, col in TARGET_COLS.items():
            row[col] = nivel1[cn]['pred']
            row[f'conf_{cn}'] = round(nivel1[cn]['conf'] * 100, 1)
            row[f'nivel_{cn}'] = 1 if nivel1[cn]['es_final'] else None
        row['necesita_nivel2'] = any(not nivel1[cn]['es_final'] for cn in TARGET_COLS)

        records.append(row)

    return pd.DataFrame(records)


def procesar_carpeta_piezas(models, folder_path=None):
    """Escanea la carpeta de piezas de 2026 (recursivo -- algunas
    universidades anidan las piezas varias carpetas adentro, ej.
    `MAR/{universidad}/{Prensa|VP}/`, bug real encontrado en la Story 1.1)
    y clasifica con Nivel 1 (SVM calibrado). No corre Nivel 2 (LLM): eso
    requiere GPU y corre en Colab -- ver `cargar_predicciones_hibridas()`
    para leer ese resultado ya calculado."""
    folder_path = folder_path or PIEZAS_DIR
    piezas = backend_ingesta.listar_piezas(folder_path)
    return _clasificar_piezas(models, piezas)


def clasificar_archivos_subidos(models, rutas_locales):
    """Clasifica piezas subidas por el usuario desde el navegador (ya
    guardadas en disco por `app.py` en una carpeta temporal) -- no depende
    de la estructura de carpetas por mes que usa `procesar_carpeta_piezas()`,
    porque una pieza suelta subida a mano no tiene esa estructura.

    `rutas_locales`: lista de rutas locales de archivos ya guardados.
    Un archivo con extensión no reconocida se omite (mismo criterio de
    robustez que `ingesta.listar_piezas()`: nunca detiene el lote).
    """
    piezas = []
    for ruta in rutas_locales:
        nombre = os.path.basename(ruta)
        ext = os.path.splitext(nombre)[1].lower()
        tipo = backend_config.EXT_MAP.get(ext)
        if tipo is None:
            continue
        anunciante, version = parse_filename(nombre)
        piezas.append({
            'archivo': nombre, 'tipo': tipo, 'ruta': ruta,
            'anunciante': anunciante, 'version': version, 'mes': '',
        })
    return _clasificar_piezas(models, piezas)


def cargar_predicciones_hibridas(path=None):
    """Lee el Excel con el pipeline híbrido COMPLETO (Nivel 1 + Nivel 2,
    Story 3.4), generado en Colab con GPU real. Esta app corre localmente
    sin GPU, así que no reejecuta el VLM/LLM: lee el resultado ya
    calculado en vez de aproximarlo con Nivel 1 solo."""
    path = path or backend_config.HIBRIDO_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró el Excel híbrido en {path}. "
            "Corré primero el escalado completo en Colab (Story 3.4, Task 3)."
        )
    return pd.read_excel(path, sheet_name='Predicciones')


def exportar_predicciones_excel(df_2026, output_path=OUTPUT_EXCEL_PATH):
    """Guarda las predicciones en un Excel estructurado."""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_2026.to_excel(writer, sheet_name='Predicciones', index=False)

        if not df_2026.empty and 'tipo_campania' in df_2026.columns:
            pivot = df_2026.pivot_table(
                index='mes', columns='tipo_campania',
                values='archivo', aggfunc='count', fill_value=0
            )
            pivot.to_excel(writer, sheet_name='Resumen_Tipo')

    return output_path


# ── Ejemplos de prueba directa si ejecutas: python -m src.predictor ────
if __name__ == "__main__":
    print("=" * 60)
    print("EJEMPLOS DE PREDICCIÓN EN CONSOLA")
    print("=" * 60)

    modelos_cargados = load_trained_models()
    predecir_pieza("MAESTRIA EN DERECHO CONSTITUCIONAL", "PONTIFICIA UNIV. CATOLICA DEL ECUADOR", models=modelos_cargados)
    predecir_pieza("MATRICULATE YA-UTPL VALIDABLE", "UNIV. TECN. PART. DE LOJA", models=modelos_cargados)
    predecir_pieza("BECAS HASTA 60 PORCIENTO APLICA YA", "UNIR", models=modelos_cargados)
    predecir_pieza("IDENTIDAD CORPORATIVA 60 ANIVERSARIO", "UNIV. CENTRAL DEL ECUADOR", models=modelos_cargados)
