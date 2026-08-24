import base64
import glob
import importlib
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from src.config import (
    GRAFICOS_DIR,
    MODEL_PATH,
    OUTPUT_EXCEL_PATH,
    PIEZAS_DIR,
    TARGET_COLS,
    TARGET_LABELS,
)
from src.predictor import (
    cargar_predicciones_hibridas,
    clasificar_archivos_subidos,
    exportar_predicciones_excel,
    load_trained_models,
    parse_filename,
    predecir_lote_dataframe,
    predecir_pieza_single,
    procesar_carpeta_piezas,
)

st.set_page_config(
    page_title="Dashboard Principal - Clasificador UIDE",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONF_LOW = 60
CONF_HIGH = 80
CAMPANIA_COLS = ["Admisión", "Posicionamiento", "Evento", "Retención"]
DIM_ROWS = [
    ("J", "Tipo de Campaña"),
    ("K", "Modalidad"),
    ("L", "Grado"),
    ("M", "Carrera"),
]


# ------------------------------------------------------------------------------
# ESTILOS
# ------------------------------------------------------------------------------
def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        [data-testid="stSidebar"] { border-right: 1px solid #123055; }
        [data-testid="stSidebarNav"] { padding-top: 0.4rem; }
        [data-testid="stSidebarNav"] span {
            font-size: 0.92rem;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e9f2;
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        [data-testid="stMetricLabel"] { font-size: 0.75rem; letter-spacing: 0.04em; }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border-color: #e5e9f2;
        }
        .hero-kicker {
            color: #64748b;
            font-size: 0.95rem;
            margin-top: -0.4rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------------
# CARGA DE MODELOS (CACHE)
# ------------------------------------------------------------------------------
@st.cache_resource
def get_models():
    try:
        return load_trained_models()
    except Exception:
        return None


def init_session() -> None:
    st.session_state.setdefault("exception_logs", [])
    st.session_state.setdefault("inference_events", [])
    st.session_state.setdefault("last_batch_df", None)
    st.session_state.setdefault("last_batch_path", None)


def log_exception(source: str, error: Exception) -> None:
    st.session_state.exception_logs.insert(
        0,
        {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fuente": source,
            "error": str(error),
        },
    )
    st.session_state.exception_logs = st.session_state.exception_logs[:80]


def record_inference(modelo: str, latency_ms: float, ok: bool = True, detalle: str = "") -> None:
    st.session_state.inference_events.append(
        {
            "ts": datetime.now(),
            "modelo": modelo,
            "latencia_ms": latency_ms,
            "ok": ok,
            "detalle": detalle,
        }
    )
    st.session_state.inference_events = st.session_state.inference_events[-200:]


# ------------------------------------------------------------------------------
# DATOS DEL DASHBOARD
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_predictions_table() -> pd.DataFrame:
    path = OUTPUT_EXCEL_PATH
    if os.path.exists(path):
        try:
            return pd.read_excel(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def inventory_files() -> pd.DataFrame:
    records = []
    if not os.path.isdir(PIEZAS_DIR):
        return pd.DataFrame()
    for root, _, files in os.walk(PIEZAS_DIR):
        for fname in files:
            if fname.startswith(("~", ".")):
                continue
            ext = os.path.splitext(fname)[1].lower()
            anunciante, version = parse_filename(fname)
            records.append(
                {
                    "archivo": fname,
                    "ruta": os.path.join(root, fname),
                    "extension": ext,
                    "tipo_archivo": {
                        ".jpg": "Imagen",
                        ".jpeg": "Imagen",
                        ".png": "Imagen",
                        ".pdf": "PDF",
                        ".mp3": "Audio",
                        ".mp4": "Video",
                    }.get(ext, ext.upper() or "Otro"),
                    "anunciante": anunciante,
                    "version": version,
                    "mes": os.path.basename(root),
                }
            )
    return pd.DataFrame(records)


def active_predictions() -> pd.DataFrame:
    if st.session_state.last_batch_df is not None and not st.session_state.last_batch_df.empty:
        return st.session_state.last_batch_df
    return load_predictions_table()


def demo_predictions() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    meses = ["ENE", "FEB", "MAR", "ABR", "MAY", "JUN"]
    tipos = CAMPANIA_COLS
    grados = ["Pregrado", "Postgrado", "Educación Continua"]
    modalidades = ["Presencial", "Híbrida", "En línea"]
    carreras = ["Derecho", "Administración", "Ingeniería", "Psicología"]
    medios = ["Imagen", "PDF", "Audio"]
    rows = []
    for i in range(28):
        tipo = tipos[i % len(tipos)]
        rows.append(
            {
                "mes": meses[i % len(meses)],
                "tipo_campania": tipo,
                "grado": grados[i % len(grados)],
                "modalidad": modalidades[i % len(modalidades)],
                "carrera": carreras[i % len(carreras)],
                "tipo_archivo": medios[i % len(medios)],
                "conf_J": float(rng.integers(48, 97)),
                "conf_K": float(rng.integers(52, 96)),
                "conf_L": float(rng.integers(61, 95)),
                "conf_M": float(rng.integers(45, 92)),
            }
        )
    return pd.DataFrame(rows)


def dashboard_frame() -> tuple[pd.DataFrame, bool]:
    df = active_predictions()
    if df is not None and not df.empty:
        return df, False
    return demo_predictions(), True


def confidence_columns(df: pd.DataFrame) -> dict[str, str]:
    mapping = {}
    for cn in TARGET_COLS:
        for candidate in (f"conf_{cn}", f"confianza_{cn}", f"conf_{TARGET_COLS[cn]}"):
            if candidate in df.columns:
                mapping[cn] = candidate
                break
    return mapping


def compute_kpis(df: pd.DataFrame, is_demo: bool) -> dict:
    conf_map = confidence_columns(df)
    conf_vals = []
    for col in conf_map.values():
        conf_vals.extend(pd.to_numeric(df[col], errors="coerce").dropna().tolist())

    baja_certeza = 0
    if conf_map:
        mask = pd.Series(False, index=df.index)
        for col in conf_map.values():
            mask = mask | (pd.to_numeric(df[col], errors="coerce") < CONF_LOW)
        baja_certeza = int(mask.sum())
    elif is_demo:
        baja_certeza = 35

    events = st.session_state.inference_events
    if events:
        lat_s = np.mean([e["latencia_ms"] for e in events[-40:]]) / 1000
    else:
        lat_s = 3.27 if is_demo else 0.0

    incidencias = int(len(st.session_state.exception_logs))
    if is_demo and incidencias == 0:
        incidencias = 15

    return {
        "activos": int(len(df)),
        "incidencias": incidencias,
        "latencia": lat_s,
        "baja_certeza": baja_certeza,
    }


def conf_color(value: float) -> str:
    if pd.isna(value):
        return "background-color: #f1f5f9; color: #64748b;"
    if value < CONF_LOW:
        return "background-color: #fecaca; color: #7f1d1d; font-weight: 600;"
    if value < CONF_HIGH:
        return "background-color: #fde68a; color: #78350f; font-weight: 600;"
    return "background-color: #bbf7d0; color: #14532d; font-weight: 600;"


def build_confidence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    conf_map = confidence_columns(df)
    camp_col = "tipo_campania" if "tipo_campania" in df.columns else None
    campaigns = (
        [c for c in df[camp_col].dropna().unique().tolist() if str(c).strip()]
        if camp_col
        else CAMPANIA_COLS
    )
    if not campaigns:
        campaigns = CAMPANIA_COLS

    matrix = pd.DataFrame(index=[label for _, label in DIM_ROWS], columns=campaigns, dtype=float)
    if not conf_map or camp_col is None:
        demo = {
            "Admisión": [62, 55, 71, 48],
            "Posicionamiento": [73, 72, 78, 67],
            "Evento": [84, 89, 85, 86],
            "Retención": [95, 66, 92, 68],
        }
        return pd.DataFrame(demo, index=[label for _, label in DIM_ROWS], dtype=float)

    for cn, label in DIM_ROWS:
        col = conf_map.get(cn)
        if not col:
            continue
        grouped = (
            df.assign(_conf=pd.to_numeric(df[col], errors="coerce"))
            .groupby(camp_col)["_conf"]
            .mean()
        )
        for camp in campaigns:
            if camp in grouped.index:
                matrix.loc[label, camp] = round(float(grouped.loc[camp]), 1)
    return matrix


def stacked_bar(df_long: pd.DataFrame, x: str, y: str, color: str, palette: list[str]) -> alt.Chart:
    return (
        alt.Chart(df_long)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X(f"{x}:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{y}:Q", stack="zero", title=None),
            color=alt.Color(
                f"{color}:N",
                scale=alt.Scale(range=palette),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=[x, color, y],
        )
        .properties(height=280)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=True, gridColor="#eef2f7", domain=False)
    )


def latency_frame() -> pd.DataFrame:
    events = st.session_state.inference_events
    if events:
        df = pd.DataFrame(events)
        df["minuto"] = pd.to_datetime(df["ts"]).dt.strftime("%H:%M:%S")
        pivot = df.pivot_table(index="minuto", columns="modelo", values="latencia_ms", aggfunc="mean")
        return pivot.tail(12).reset_index()

    rng = np.random.default_rng(int(time.time()) // 4)
    ticks = [f"t{i}" for i in range(8)]
    return pd.DataFrame(
        {
            "minuto": ticks,
            "Qwen2.5-VL": rng.integers(1900, 3800, size=8),
            "faster-whisper": rng.integers(1100, 1800, size=8),
            "TF-IDF": np.concatenate([rng.integers(180, 260, size=6), rng.integers(1800, 2300, size=2)]),
        }
    )


# ------------------------------------------------------------------------------
# VISTAS
# ------------------------------------------------------------------------------
def page_dashboard() -> None:
    models = get_models()
    df, is_demo = dashboard_frame()
    kpis = compute_kpis(df, is_demo)

    head_l, head_r = st.columns([4, 1.2])
    with head_l:
        st.title("Dashboard Principal")
        st.markdown(
            '<p class="hero-kicker">Ingesta, análisis y visualización centralizada de piezas publicitarias institucionales y multicanal.</p>',
            unsafe_allow_html=True,
        )
    with head_r:
        st.write("")
        st.write("")
        if st.button("Cargar nuevos activos", type="primary", use_container_width=True):
            # Con callables, switch_page exige el objeto st.Page (no el url_path).
            st.switch_page(PAGE_CARGA_MASIVA)

    if is_demo:
        st.caption("Mostrando vista de referencia. Procesa un lote para sustituir los indicadores con inferencia real.")
    elif models is None:
        st.warning("Los artefactos `.pkl` no están cargados. Entrena el pipeline en Estado de Pipelines.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ACTIVOS PROCESADOS", f"{kpis['activos']}", "Inferencia completada")
    m2.metric("INCIDENCIAS", f"{kpis['incidencias']}", "Corruptos o fallidos", delta_color="inverse")
    m3.metric("LATENCIA MEDIA", f"{kpis['latencia']:.2f} s", "Todos los modelos")
    m4.metric("BAJA CERTEZA", f"{kpis['baja_certeza']}", "Requieren revisión humana")

    st.write("")
    left, right = st.columns([1.25, 1])

    with left:
        with st.container(border=True):
            st.subheader("Distribución de campañas por dimensión")
            st.caption("Jerarquía Tipo de Campaña · Grado · Modalidad · Carrera.")
            if "tipo_campania" in df.columns and "grado" in df.columns:
                dist = (
                    df.groupby(["tipo_campania", "grado"], dropna=False)
                    .size()
                    .reset_index(name="activos")
                )
                chart = stacked_bar(
                    dist,
                    x="tipo_campania",
                    y="activos",
                    color="grado",
                    palette=["#1d4ed8", "#38bdf8", "#0f172a"],
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                c1, c2, c3 = st.columns(3)
                c1.info("**Pregrado**\n\nVolumen alto")
                c2.success("**Postgrado**\n\nEstable")
                c3.warning("**Educación Continua**\n\nEn crecimiento")

    with right:
        with st.container(border=True):
            st.subheader("Ratio de activos por modalidad de medios")
            st.caption("Volumen de Texto/OCR (PDF), Visión (JPG/PNG) y Audio (MP3).")
            media_col = "tipo_archivo" if "tipo_archivo" in df.columns else None
            mes_col = "mes" if "mes" in df.columns else None
            if media_col:
                if mes_col:
                    media = df.groupby([mes_col, media_col]).size().reset_index(name="activos")
                    x_field = mes_col
                else:
                    media = df.groupby(media_col).size().reset_index(name="activos")
                    media.insert(0, "fuente", "Lote")
                    x_field = "fuente"
                keep = media[media[media_col].isin(["Imagen", "PDF", "Audio", "Video"])]
                if keep.empty:
                    keep = media
                chart = stacked_bar(
                    keep,
                    x=x_field,
                    y="activos",
                    color=media_col,
                    palette=["#2563eb", "#7dd3fc", "#1e3a5f", "#94a3b8"],
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                chart_data = pd.DataFrame(
                    {
                        "mes": ["Ene", "Feb", "Mar", "Abr", "May", "Jun"],
                        "Imagen": [40, 55, 65, 58, 60, 45],
                        "PDF": [25, 30, 35, 32, 30, 25],
                        "Audio": [15, 20, 18, 22, 19, 15],
                    }
                ).melt("mes", var_name="medio", value_name="activos")
                st.altair_chart(
                    stacked_bar(chart_data, "mes", "activos", "medio", ["#2563eb", "#7dd3fc", "#1e3a5f"]),
                    use_container_width=True,
                )

    st.write("")
    bot_l, bot_r = st.columns(2)
    with bot_l:
        with st.container(border=True):
            st.subheader("Matriz de confianza de clasificación NLP")
            st.caption("Certeza promedio del modelo TF-IDF por dimensión. Rojo < 60% · Ámbar 60–80% · Verde ≥ 80%.")
            matrix = build_confidence_matrix(df)
            styler = matrix.style.format("{:.0f}%", na_rep="—")
            if hasattr(styler, "map"):
                styled = styler.map(conf_color)
            else:
                styled = styler.applymap(conf_color)
            st.dataframe(styled, use_container_width=True, height=260)

    with bot_r:
        render_latency_monitor()


def render_latency_monitor() -> None:
    with st.container(border=True):
        st.subheader("Monitor de latencia e inferencia MLOps")
        st.caption("Tiempos de ejecución por modelo en la infraestructura (ms).")
        lat_df = latency_frame()
        long = lat_df.melt("minuto", var_name="modelo", value_name="ms")
        chart = (
            alt.Chart(long)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("minuto:N", title=None),
                y=alt.Y("ms:Q", title="ms"),
                color=alt.Color(
                    "modelo:N",
                    scale=alt.Scale(range=["#2563eb", "#0ea5e9", "#f59e0b"]),
                    legend=alt.Legend(orient="bottom", title=None),
                ),
                tooltip=["minuto", "modelo", "ms"],
            )
            .properties(height=260)
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(chart, use_container_width=True)
        if st.session_state.inference_events:
            last = st.session_state.inference_events[-1]
            st.caption(
                f"Última inferencia: {last['modelo']} · {last['latencia_ms']:.0f} ms · "
                f"{last['ts'].strftime('%H:%M:%S')}"
            )


def _filtrar_columnas_seleccionadas(df: pd.DataFrame, columnas_sel: list[str]) -> pd.DataFrame:
    """Oculta del resultado las dimensiones que el usuario no seleccionó.
    El modelo igual clasifica las 4 (es rápido y no vale la pena complicar
    el pipeline para ahorrar ese cómputo) -- esto solo filtra qué se
    muestra y se exporta, según lo que el usuario pidió ver."""
    columnas_omitir = [cn for cn in TARGET_COLS if cn not in columnas_sel]
    drop_cols = []
    for cn in columnas_omitir:
        col = TARGET_COLS[cn]
        drop_cols += [c for c in (col, f"conf_{cn}", f"nivel_{cn}") if c in df.columns]
    return df.drop(columns=drop_cols) if drop_cols else df


def _agregar_miniaturas(df: pd.DataFrame, col_ruta: str = "ruta_local") -> pd.DataFrame:
    """Agrega una columna "miniatura" al principio de la tabla con la imagen
    de cada pieza (solo tipo Imagen -- audio/video/PDF no tienen una vista
    previa estática que tenga sentido). Se muestra con
    `st.column_config.ImageColumn`, que ya trae el "click para ampliar"
    integrado -- no hace falta programarlo aparte."""
    if "tipo_archivo" not in df.columns or col_ruta not in df.columns:
        return df

    def _thumb(row):
        if row.get("tipo_archivo") != "Imagen":
            return None
        ruta = row.get(col_ruta)
        if not ruta or not os.path.exists(ruta):
            return None
        try:
            with open(ruta, "rb") as f:
                datos = f.read()
            ext = os.path.splitext(ruta)[1].lower().lstrip(".")
            mime = "jpeg" if ext == "jpg" else ext
            return f"data:image/{mime};base64,{base64.b64encode(datos).decode('ascii')}"
        except Exception:
            return None

    df = df.copy()
    df.insert(0, "miniatura", df.apply(_thumb, axis=1))
    return df


def page_carga_masiva() -> None:
    models = get_models()
    st.title("Carga masiva")
    st.caption("Elegí el modelo, subí tus piezas o un Excel, y generá las predicciones.")

    if models is None:
        st.error("Modelos no cargados. Entrene el pipeline en Estado de Pipelines.")

    st.write("")
    col_modelo, col_cols = st.columns([1.3, 2])
    with col_modelo:
        modo = st.radio(
            "Modelo a usar",
            ["rapido", "hibrido"],
            format_func=lambda k: {
                "rapido": "⚡ Clasificar mis piezas ahora (rápido)",
                "hibrido": "✅ Ver resultados completos (ya procesados con IA)",
            }[k],
            help=(
                "\"Clasificar ahora\" analiza las piezas que subas en el momento, en segundos. "
                "\"Ver resultados completos\" muestra las 1.414 piezas que ya se clasificaron antes "
                "con una inteligencia artificial más avanzada, en un servidor con más capacidad -- "
                "en este modo no se pueden subir piezas nuevas, porque esta app no tiene el "
                "hardware necesario para correr esa IA en el momento."
            ),
        )
    with col_cols:
        columnas_sel = st.multiselect(
            "Dimensiones a clasificar",
            options=list(TARGET_COLS.keys()),
            default=list(TARGET_COLS.keys()),
            format_func=lambda cn: TARGET_LABELS[cn],
            help="Las columnas que no selecciones no se muestran ni se exportan.",
        )
    if not columnas_sel:
        st.warning("Seleccioná al menos una dimensión para clasificar.")
        return

    with st.expander("¿Querés clasificar por una columna nueva (ej. \"ciudad\")? Así se agrega"):
        st.markdown(
            "Estas 4 dimensiones (J/K/L/M) ya tienen años de datos históricos, por eso se "
            "clasifican solas y al instante acá. Una columna nueva no tiene ese historial, así "
            "que no se puede entrenar un modelo rápido para ella -- se agrega con estos pasos:\n\n"
            "**a.** Alguien del equipo define la columna nueva en Colab: su nombre (ej. "
            "\"ciudad\") y la lista cerrada de valores válidos (ej. Quito, Guayaquil, Cuenca...) "
            "-- así la IA nunca puede inventar un valor que no está en la lista.\n\n"
            "**b.** Se corre la clasificación de esa columna sobre las piezas, en Colab (necesita "
            "GPU, por eso no se hace desde acá).\n\n"
            "**c.** Se exporta el resultado a un Excel, con la columna nueva agregada junto a J/K/L/M.\n\n"
            "**d.** Ese Excel se sube en \"Ver resultados completos\" → \"¿Necesitás clasificar "
            "piezas nuevas con IA?\" -- la columna nueva va a aparecer en la tabla sin que haga "
            "falta ningún cambio más acá."
        )

    st.write("")

    if modo == "hibrido":
        with st.container(border=True):
            st.subheader("Resultados completos, ya procesados con IA")
            st.caption(
                "Estas 1.414 piezas ya fueron analizadas antes con una inteligencia artificial más "
                "avanzada, en un servidor con más capacidad. Acá podés consultar esos resultados, "
                "pero en este modo no se pueden subir piezas nuevas -- para eso usá "
                "\"Clasificar mis piezas ahora\"."
            )
            if st.button("Ver resultados completos", type="primary"):
                try:
                    df_hibrido = _filtrar_columnas_seleccionadas(cargar_predicciones_hibridas(), columnas_sel)
                    st.session_state.last_batch_df = df_hibrido
                    st.session_state.last_batch_path = None
                    st.cache_data.clear()
                    st.success(f"Resultados cargados: {len(df_hibrido)} piezas.")
                    st.dataframe(df_hibrido.head(20), use_container_width=True)
                except FileNotFoundError as exc:
                    st.warning(str(exc))

        with st.expander("¿Necesitás clasificar piezas nuevas con IA (más preciso)?"):
            st.markdown(
                "Esta app pública no tiene la GPU que hace falta para correr la IA, así que ese "
                "paso se hace aparte, en Google Colab (que sí presta GPU gratis). Son 3 pasos:\n\n"
                "1. Un miembro del equipo abre el notebook del proyecto en Google Colab y lo corre "
                "con GPU sobre las piezas nuevas -- esto genera un Excel con la clasificación "
                "completa (ML + IA).\n"
                "2. Descargá ese Excel de Colab.\n"
                "3. Subilo acá abajo para verlo en esta app, sin esperar a que alguien actualice "
                "la app entera."
            )
            subido_hibrido = st.file_uploader(
                "Subir Excel ya clasificado con IA (de Colab)", type=["xlsx"], key="upload_hibrido"
            )
            if subido_hibrido is not None:
                try:
                    df_nuevo = pd.read_excel(subido_hibrido, sheet_name="Predicciones")
                    df_nuevo = _filtrar_columnas_seleccionadas(df_nuevo, columnas_sel)
                    st.session_state.last_batch_df = df_nuevo
                    st.session_state.last_batch_path = None
                    st.success(f"Resultados nuevos cargados: {len(df_nuevo)} piezas.")
                    st.dataframe(df_nuevo.head(20), use_container_width=True)
                except Exception as exc:
                    st.error(
                        f"No se pudo leer el archivo (¿tiene una hoja llamada 'Predicciones'?): {exc}"
                    )
        return

    tab_subir, tab_carpeta, tab_excel = st.tabs(
        ["📤 Subir piezas", "📁 Escanear carpeta del servidor", "📄 Subir Excel/CSV"]
    )

    with tab_subir:
        st.caption(
            "**Usá esta opción si tenés los archivos de las piezas en tu computadora** (imágenes, "
            "PDF, audio o video). Subilos acá y se clasifican al toque. Importante: cada nombre de "
            "archivo debe seguir el formato `ANUNCIANTE - VERSIÓN(FORMATO).ext` (ej. `PUCE - "
            "MAESTRIA EN DERECHO(COMERCIAL).jpg`) -- de ahí se saca automáticamente el anunciante y "
            "la versión de la pieza."
        )
        archivos_subidos = st.file_uploader(
            "Piezas a clasificar",
            type=["jpg", "jpeg", "png", "pdf", "mp3", "mp4", "wmv", "gif"],
            accept_multiple_files=True,
        )
        if archivos_subidos and st.button("Clasificar piezas subidas", type="primary", disabled=models is None):
            try:
                with st.spinner(f"Clasificando {len(archivos_subidos)} piezas..."):
                    tmp_dir = tempfile.mkdtemp(prefix="piezas_subidas_")
                    rutas_locales = []
                    for archivo in archivos_subidos:
                        ruta = os.path.join(tmp_dir, archivo.name)
                        with open(ruta, "wb") as f:
                            f.write(archivo.getbuffer())
                        rutas_locales.append(ruta)

                    t0 = time.perf_counter()
                    df_subidas = clasificar_archivos_subidos(models, rutas_locales)
                    elapsed = (time.perf_counter() - t0) * 1000
                record_inference("TF-IDF", elapsed, True, f"{len(df_subidas)} piezas subidas")
                if df_subidas.empty:
                    st.warning("Ninguno de los archivos subidos tiene una extensión reconocida.")
                else:
                    df_subidas = _filtrar_columnas_seleccionadas(df_subidas, columnas_sel)
                    # La miniatura se arma ANTES de tirar ruta_local (la
                    # necesita para leer el archivo real) -- ruta_local en sí
                    # apunta a una carpeta temporal que se borra al cerrar la
                    # sesión, no tiene sentido mostrarla ni exportarla.
                    df_subidas_vista = _agregar_miniaturas(df_subidas).drop(columns=["ruta_local"])
                    df_subidas = df_subidas.drop(columns=["ruta_local"])
                    st.session_state.last_batch_df = df_subidas
                    st.session_state.last_batch_path = None
                    st.cache_data.clear()
                    st.success(f"Clasificación completada: {len(df_subidas)} piezas.")
                    st.dataframe(
                        df_subidas_vista,
                        use_container_width=True,
                        column_config={
                            "miniatura": st.column_config.ImageColumn("Vista previa", width="small")
                        },
                    )
            except Exception as exc:
                log_exception("Carga masiva / piezas subidas", exc)
                record_inference("TF-IDF", 0, False, str(exc))
                st.error(str(exc))

    with tab_carpeta:
        st.caption(
            "**Usá esta opción si las piezas ya están guardadas en el servidor de la app** (no en "
            "tu computadora) -- no hace falta subir nada, solo tocar el botón y procesa todo lo que "
            "encuentre en esa carpeta de una sola vez. Carpeta que va a escanear: "
            f"`{PIEZAS_DIR}`. Si una pieza es de las que necesitan la revisión de IA (Nivel 2, más "
            "compleja de clasificar), acá queda marcada para revisar aparte -- esta opción solo usa "
            "el modelo rápido (ML)."
        )
        if st.button("Procesar carpeta completa", type="primary", disabled=models is None):
            try:
                with st.spinner("Procesando archivos e invocando `procesar_carpeta_piezas`..."):
                    t0 = time.perf_counter()
                    df_2026 = procesar_carpeta_piezas(models, folder_path=PIEZAS_DIR)
                    elapsed = (time.perf_counter() - t0) * 1000
                record_inference("TF-IDF", elapsed, True, f"{len(df_2026)} archivos")
                if df_2026.empty:
                    st.warning("No se encontraron archivos válidos.")
                else:
                    output_file = exportar_predicciones_excel(df_2026, output_path=OUTPUT_EXCEL_PATH)
                    df_2026 = _filtrar_columnas_seleccionadas(df_2026, columnas_sel)
                    st.session_state.last_batch_df = df_2026
                    st.session_state.last_batch_path = str(output_file)
                    st.cache_data.clear()
                    st.success(f"Procesamiento completado: {len(df_2026)} registros.")
                    # Miniaturas solo sobre las primeras 20 filas que se
                    # muestran -- generarlas para las 1.414 piezas completas
                    # sería lento y no se ven todas a la vez igual.
                    st.dataframe(
                        _agregar_miniaturas(df_2026.head(20)),
                        use_container_width=True,
                        column_config={
                            "miniatura": st.column_config.ImageColumn("Vista previa", width="small")
                        },
                    )
                    with open(output_file, "rb") as file:
                        st.download_button(
                            "Descargar Excel de predicciones",
                            data=file,
                            file_name="Predicciones_2026_ML.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
            except Exception as exc:
                log_exception("Carga masiva / carpeta", exc)
                record_inference("TF-IDF", 0, False, str(exc))
                st.error(str(exc))

    with tab_excel:
        st.caption(
            "**Usá esta opción si NO tenés los archivos de las piezas, pero sí tenés una planilla** "
            "con la lista (por ejemplo, exportada de otro sistema) -- ahorra tener que subir "
            "archivo por archivo. La planilla tiene que tener, como mínimo, una columna "
            "`anunciante` y una columna `version` con el texto de cada pieza."
        )
        uploaded = st.file_uploader("Planilla con las piezas a clasificar (.xlsx o .csv)", type=["xlsx", "csv"])
        if uploaded is not None:
            df_upload = (
                pd.read_excel(uploaded)
                if uploaded.name.endswith(".xlsx")
                else pd.read_csv(uploaded)
            )
            st.dataframe(df_upload.head(5), use_container_width=True)
            if st.button("Ejecutar predicciones en carga", disabled=models is None):
                if "version" not in df_upload.columns or "anunciante" not in df_upload.columns:
                    st.error("El archivo debe contener las columnas `version` y `anunciante`.")
                else:
                    try:
                        t0 = time.perf_counter()
                        df_preds = _filtrar_columnas_seleccionadas(predecir_lote_dataframe(models, df_upload), columnas_sel)
                        elapsed = (time.perf_counter() - t0) * 1000
                        record_inference("TF-IDF", elapsed, True, f"{len(df_preds)} filas")
                        st.session_state.last_batch_df = df_preds
                        st.dataframe(df_preds, use_container_width=True)
                    except Exception as exc:
                        log_exception("Carga masiva / archivo", exc)
                        st.error(str(exc))


def page_monitor_extraccion() -> None:
    st.title("Monitor de extracción")
    st.caption("Parseo de nombres de archivo y cobertura de la carpeta multi-modal.")
    inv = inventory_files()
    if inv.empty:
        st.info(f"No hay archivos en `{PIEZAS_DIR}`.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Archivos detectados", len(inv))
    c2.metric("Con anunciante", int((inv["anunciante"].astype(str).str.len() > 0).sum()))
    c3.metric("Formatos", inv["tipo_archivo"].nunique())
    st.dataframe(inv, use_container_width=True, hide_index=True)


def page_matriz() -> None:
    models = get_models()
    st.title("Matriz de clasificación")
    st.caption("Inferencia individual sobre las dimensiones J, K, L y M.")

    col1, col2 = st.columns(2)
    with col1:
        version_input = st.text_input("Versión / texto de la pieza", value="MAESTRIA EN DERECHO CONSTITUCIONAL")
        anunciante_input = st.text_input("Anunciante / universidad", value="PONTIFICIA UNIV. CATOLICA DEL ECUADOR")
    with col2:
        file_name_parser = st.text_input(
            "O simular desde nombre de archivo",
            value="PONTIFICIA UNIV. CATOLICA DEL ECUADOR - MAESTRIA EN DERECHO CONSTITUCIONAL(1080x1080)-FB.mp4",
        )
        if st.button("Extraer datos del nombre"):
            anunciante_parsed, version_parsed = parse_filename(file_name_parser)
            st.info(f"**Anunciante:** {anunciante_parsed}\n\n**Versión:** {version_parsed}")

    if st.button("Clasificar activo", type="primary"):
        if models is None:
            st.error("Los modelos no están cargados. Entrene el pipeline primero.")
            return
        try:
            t0 = time.perf_counter()
            resultados = predecir_pieza_single(models, version=version_input, anunciante=anunciante_input)
            elapsed = (time.perf_counter() - t0) * 1000
            record_inference("TF-IDF", elapsed, True, version_input[:40])
            cols = st.columns(len(TARGET_COLS))
            for i, cn in enumerate(TARGET_COLS):
                r = resultados[cn]
                with cols[i]:
                    st.metric(r["label"], r["pred"], f"{r['conf'] * 100:.1f}% confianza")
                    st.progress(min(float(r["conf"]), 1.0))
                    for cls, p in r["top3"]:
                        st.caption(f"{cls}: {p * 100:.1f}%")
        except Exception as exc:
            log_exception("Matriz de clasificación", exc)
            st.error(str(exc))


def page_revision_manual() -> None:
    st.title("Revisión manual")
    st.caption("Cola de governance para predicciones con baja certeza.")
    df, is_demo = dashboard_frame()
    conf_map = confidence_columns(df)
    if not conf_map:
        st.info("Aún no hay columnas de confianza. Ejecute una carga masiva.")
        return

    umbral = st.slider("Umbral de certeza mínima (%)", 40, 90, CONF_LOW)
    mask = pd.Series(False, index=df.index)
    for col in conf_map.values():
        mask = mask | (pd.to_numeric(df[col], errors="coerce") < umbral)
    cola = df.loc[mask]
    st.metric("Casos a revisar", len(cola))
    if is_demo:
        st.caption("Cola construida sobre la vista de referencia.")
    st.dataframe(cola, use_container_width=True)


def page_explorador() -> None:
    st.title("Explorador de medios")
    st.caption("Biblioteca de activos en disco, agrupada por tipo de archivo.")
    inv = inventory_files()
    if inv.empty:
        st.info(f"No hay medios en `{PIEZAS_DIR}`.")
        return
    tipo = st.multiselect("Filtrar modalidad", sorted(inv["tipo_archivo"].unique()), default=None)
    view = inv if not tipo else inv[inv["tipo_archivo"].isin(tipo)]
    st.dataframe(view, use_container_width=True, hide_index=True)


def page_exportacion() -> None:
    st.title("Exportación de reportes")
    st.caption("Descarga del Excel de predicciones generado por el lote.")
    path = st.session_state.last_batch_path or (str(OUTPUT_EXCEL_PATH) if os.path.exists(OUTPUT_EXCEL_PATH) else None)
    df = active_predictions()
    if df is None or df.empty:
        st.warning("No hay predicciones para exportar. Ejecute Carga masiva.")
        return
    st.dataframe(df.head(30), use_container_width=True)
    if path and os.path.exists(path):
        with open(path, "rb") as file:
            st.download_button(
                "Descargar Predicciones_2026_ML.xlsx",
                data=file,
                file_name="Predicciones_2026_ML.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )
    else:
        st.info("El archivo Excel aún no está en disco. Vuelva a procesar el lote para materializarlo.")


def page_pipelines() -> None:
    models = get_models()
    st.title("Estado de pipelines")
    st.caption("Salud del artefacto ML y reentrenamiento del clasificador TF-IDF.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Modelo en memoria", "OK" if models is not None else "No cargado")
    c2.metric("Artefacto .pkl", "Sí" if os.path.exists(MODEL_PATH) else "No")
    if HAS_PSUTIL:
        c3.metric("CPU host", f"{psutil.cpu_percent()}%")
    else:
        c3.metric("psutil", "No instalado")

    st.subheader("Reentrenamiento")
    st.info("Carga el histórico, vectoriza con TF-IDF y reentrena las canalizaciones J, K, L y M.")
    if st.button("Iniciar entrenamiento completo", type="primary"):
        log_container = st.empty()
        try:
            with st.spinner("Entrenando modelos y generando gráficos..."):
                import src.trainer as trainer

                importlib.reload(trainer)

                class StreamToST:
                    def __init__(self, container):
                        self.container = container
                        self.text = ""

                    def write(self, s):
                        self.text += s
                        self.container.code(self.text, language="text")

                    def flush(self):
                        pass

                sys.stdout = StreamToST(log_container)
                trainer.main()
                sys.stdout = sys.__stdout__
                st.cache_resource.clear()
                st.success("Entrenamiento completado. Recargue el dashboard para usar el nuevo pipeline.")
        except Exception as exc:
            sys.stdout = sys.__stdout__
            log_exception("Entrenamiento", exc)
            st.error(f"Error durante el entrenamiento: {exc}")

    if os.path.exists(GRAFICOS_DIR):
        images = sorted(glob.glob(os.path.join(GRAFICOS_DIR, "*.png")))
        for img in images:
            st.image(img, caption=os.path.basename(img), use_container_width=True)


def page_logs() -> None:
    st.title("Logs de excepciones")
    st.caption("Bitácora de fallos de inferencia, I/O y entrenamiento en esta sesión.")
    logs = st.session_state.exception_logs
    if not logs:
        st.success("Sin excepciones registradas en la sesión actual.")
        return
    st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------
# ROUTER
# ------------------------------------------------------------------------------
inject_styles()
init_session()

if HAS_PSUTIL:
    ram = psutil.virtual_memory()
    st.sidebar.caption(f"CPU {psutil.cpu_percent()}% · RAM {ram.percent:.0f}%")
else:
    st.sidebar.caption("Clasificador UIDE · MLOps")

PAGE_CARGA_MASIVA = st.Page(
    page_carga_masiva, title="Carga Masiva", icon="📤", url_path="carga-masiva"
)

pages = {
    "GENERAL": [
        st.Page(page_dashboard, title="Dashboard Principal", icon="📊", default=True),
    ],
    "INGESTA MULTI-MODAL": [
        PAGE_CARGA_MASIVA,
        st.Page(page_monitor_extraccion, title="Monitor de Extracción", icon="📡"),
    ],
    "CLASIFICACIÓN & GOVERNANCE": [
        st.Page(page_matriz, title="Matriz de Clasificación", icon="🗂️"),
        st.Page(page_revision_manual, title="Revisión Manual", icon="🧑‍⚖️"),
    ],
    "BIBLIOTECA DE ACTIVOS": [
        st.Page(page_explorador, title="Explorador de Medios", icon="🖼️"),
        st.Page(page_exportacion, title="Exportación de Reportes", icon="📥"),
    ],
    "MLOPS & SYSTEM HEALTH": [
        st.Page(page_pipelines, title="Estado de Pipelines", icon="⚙️"),
        st.Page(page_logs, title="Logs de Excepciones", icon="🪵"),
    ],
}

pg = st.navigation(pages, position="sidebar")
pg.run()
