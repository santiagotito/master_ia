import os
import glob
import sys
# Agregar la carpeta raíz del proyecto al sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import streamlit as st

# Manejo seguro de psutil
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from src.config import (
    TARGET_COLS,
    TARGET_LABELS,
    GRAFICOS_DIR,
    OUTPUT_EXCEL_PATH,
    PIEZAS_DIR
)
from src.predictor import (
    load_trained_models,
    predecir_pieza_single,
    predecir_lote_dataframe,
    procesar_carpeta_piezas,
    exportar_predicciones_excel,
    parse_filename
)

# ------------------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Clasificador de Piezas Publicitarias - UIDE",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 Clasificador Multietiqueta de Piezas Publicitarias")
st.markdown("Sistema inteligente de inferencia, entrenamiento y diagnóstico de modelos de ML para clasificación de campañas, modalidades, grados y carreras.")

# ------------------------------------------------------------------------------
# CARGA DE MODELOS (CACHE)
# ------------------------------------------------------------------------------
@st.cache_resource
def get_models():
    try:
        return load_trained_models()
    except Exception as e:
        st.warning(f"No se pudieron cargar modelos previo a entrenamiento: {e}")
        return None

models = get_models()

# ------------------------------------------------------------------------------
# SIDEBAR: RECURSOS DEL SISTEMA Y CONFIGURACIÓN
# ------------------------------------------------------------------------------
st.sidebar.header("⚙️ Estado del Sistema")

if HAS_PSUTIL:
    cpu_usage = psutil.cpu_percent()
    ram_info = psutil.virtual_memory()
    st.sidebar.metric("Uso de CPU", f"{cpu_usage}%")
    st.sidebar.metric(
        "Uso de RAM", 
        f"{ram_info.percent}%", 
        f"{(ram_info.used / (1024**3)):.1f} GB / {(ram_info.total / (1024**3)):.1f} GB"
    )
else:
    st.sidebar.info("📌 Instale `psutil` para ver el estado de CPU/RAM.")

st.sidebar.divider()
st.sidebar.markdown("**Módulos disponibles:**")
st.sidebar.info("• Inferencia Individual\n• Procesamiento en Lote\n• Entrenamiento de Modelos\n• Métricas y Gráficos")

# ------------------------------------------------------------------------------
# PESTAÑAS PRINCIPALES DE LA APLICACIÓN
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Inferencia Individual", 
    "📁 Clasificación en Lote (2026)", 
    "⚙️ Entrenamiento de Modelos",
    "📊 Diagnóstico y Gráficos de Evaluación"
])

# ==============================================================================
# TAB 1: INFERENCIA INDIVIDUAL
# ==============================================================================
with tab1:
    st.subheader("Predicción rápida de una pieza publicitaria")
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        version_input = st.text_input("Versión / Texto de la pieza:", value="MAESTRIA EN DERECHO CONSTITUCIONAL")
        anunciante_input = st.text_input("Anunciante / Universidad:", value="PONTIFICIA UNIV. CATOLICA DEL ECUADOR")
    
    with col_input2:
        file_name_parser = st.text_input("O simular desde nombre de archivo:", value="PONTIFICIA UNIV. CATOLICA DEL ECUADOR - MAESTRIA EN DERECHO CONSTITUCIONAL(1080x1080)-FB.mp4")
        if st.button("Extraer datos del nombre de archivo"):
            anunciante_parsed, version_parsed = parse_filename(file_name_parser)
            st.info(f"**Anunciante extraído:** {anunciante_parsed}\n\n**Versión extraída:** {version_parsed}")

    if st.button("🚀 Clasificar Pieza", type="primary"):
        if models is None:
            st.error("Los modelos no se han cargado correctamente. Entrene el modelo primero en la pestaña 'Entrenamiento de Modelos'.")
        else:
            resultados = predecir_pieza_single(models, version=version_input, anunciante=anunciante_input)
            
            st.markdown("---")
            st.subheader("Resultados de la Predicción")
            
            cols = st.columns(len(TARGET_COLS))
            for i, (cn, col_name) in enumerate(TARGET_COLS.items()):
                r = resultados[cn]
                with cols[i]:
                    st.metric(
                        label=r['label'], 
                        value=r['pred'], 
                        delta=f"{r['conf']*100:.1f}% Confianza"
                    )
                    st.progress(min(r['conf'], 1.0))
                    
                    st.write("**Top 3 probabilidades:**")
                    for cls, p in r['top3']:
                        st.caption(f"• **{cls}**: {p*100:.1f}%")

# ==============================================================================
# TAB 2: PROCESAMIENTO EN LOTE (2026)
# ==============================================================================
with tab2:
    st.subheader("Clasificación en lote y exportación a Excel")
    
    col_batch1, col_batch2 = st.columns(2)
    
    with col_batch1:
        st.markdown("### Opción A: Escanear carpeta de datos")
        st.caption(f"Carpeta configurada: `{PIEZAS_DIR}`")
        if st.button("⚡ Procesar Carpeta Completa 2026"):
            if models is not None:
                with st.spinner("Procesando y clasificando archivos por mes..."):
                    df_2026 = procesar_carpeta_piezas(models, folder_path=PIEZAS_DIR)
                    if not df_2026.empty:
                        output_file = exportar_predicciones_excel(df_2026, output_path=OUTPUT_EXCEL_PATH)
                        st.success(f"¡Procesamiento completado! Se analizaron {len(df_2026)} registros.")
                        st.dataframe(df_2026.head(20), use_container_width=True)
                        
                        with open(output_file, "rb") as file:
                            st.download_button(
                                label="📥 Descargar Excel de Predicciones",
                                data=file,
                                file_name="Predicciones_2026_ML.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    else:
                        st.warning("No se encontraron archivos válidos en las subcarpetas.")
            else:
                st.error("Modelos no cargados. Ejecute el entrenamiento primero.")
    
    with col_batch2:
        st.markdown("### Opción B: Cargar archivo Excel/CSV")
        uploaded_file = st.file_uploader("Subir dataset para inferencia", type=['xlsx', 'csv'])
        if uploaded_file is not None:
            df_upload = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
            st.write("Vista previa de los datos:", df_upload.head(3))
            
            if st.button("Ejecutar Predicciones en Carga"):
                if models is not None:
                    if 'version' in df_upload.columns and 'anunciante' in df_upload.columns:
                        df_preds = predecir_lote_dataframe(models, df_upload)
                        st.dataframe(df_preds, use_container_width=True)
                    else:
                        st.error("El archivo debe contener las columnas 'version' y 'anunciante'.")
                else:
                    st.error("Modelos no cargados. Ejecute el entrenamiento primero.")

# ==============================================================================
# TAB 3: ENTRENAMIENTO DEL MODELO
# ==============================================================================
with tab3:
    st.subheader("🛠️ Reentrenamiento del Modelo de Machine Learning")
    st.markdown("Ejecute el pipeline de reentrenamiento completo llamando al módulo `src.trainer`.")

    col_train1, col_train2 = st.columns([2, 1])

    with col_train1:
        st.info("Este proceso cargará el dataset histórico, procesará los textos con TF-IDF, reentrenará las canalizaciones para cada variable objetivo y guardará el pipeline `.pkl` en la carpeta `models/`.")
        
        if st.button("🚀 Iniciar Entrenamiento Completo", type="primary"):
            st.markdown("### 📋 Progreso y Logs del Entrenamiento:")
            log_container = st.empty()
            
            try:
                with st.spinner("Entrenando modelos y generando gráficos... Por favor espere."):
                    import importlib
                    import src.trainer as trainer
                    importlib.reload(trainer)  # Forzar la recarga con el sys.path actualizado
                    
                    # Redirigir stdout para capturar la salida de consola en Streamlit
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
                    
                    # 1. Ejecutar la función principal del entrenador
                    if hasattr(trainer, 'main'):
                        trainer.main()
                    elif hasattr(trainer, 'train_models'):
                        trainer.train_models()
                    else:
                        st.error("No se encontró la función `main()` o `train_models()` en `src/trainer.py`.")

                    # 2. Restaurar salida estándar
                    sys.stdout = sys.__stdout__
                    
                    # 3. Recargar modelos en caché de Streamlit
                    st.cache_resource.clear()
                    models = get_models()
                    
                    st.success("✅ ¡Entrenamiento completado exitosamente y modelos actualizados!")
            
            except Exception as e:
                sys.stdout = sys.__stdout__
                st.error(f"❌ Ocurrió un error durante el entrenamiento: {e}")

        st.markdown("---")
        st.subheader("🖼️ Resultados y Gráficos de Evaluación Generados")

        # Cargar dinámicamente los gráficos 01 a 07 guardados en la carpeta de resultados
        from src.config import GRAFICOS_DIR
        
        if os.path.exists(GRAFICOS_DIR):
            archivos_graficos = sorted(glob.glob(os.path.join(GRAFICOS_DIR, "*.png")))
            
            if archivos_graficos:
                for img_path in archivos_graficos:
                    nombre_archivo = os.path.basename(img_path)
                    st.markdown(f"#### 📈 {nombre_archivo}")
                    st.image(img_path, use_container_width=True)
            else:
                st.info("ℹ️ Aún no hay gráficos en la carpeta de resultados. Ejecute el entrenamiento para generarlos.")
        else:
            st.warning("⚠️ No se encontró la carpeta de gráficos configurada.")

    with col_train2:
        st.markdown("### ℹ️ Detalles del Pipeline")
        st.write("- **Extracción de características:** TF-IDF Vectorizer")
        st.write("- **Clasificadores:** Multi-output / Independientes para J, K, L, M")
        st.write("- **Métricas evaluadas:** Accuracy, F1-Score, Precision, Recall")
        st.write("- **Gráficos esperados:** 01 al 07 en `GRAFICOS_DIR`")

# ==============================================================================
# TAB 4: DIAGNÓSTICO Y GRÁFICOS DE EVALUACIÓN
# ==============================================================================
with tab4:
    st.subheader("Gráficos y Estadísticas de Evaluación del Modelo")
    st.markdown("Diagnóstico cualitativo y cuantitativo extraído desde `evaluador.py`.")
    
    # Mapeo de gráficos
    graficos_archivos = {
        "01_distribucion_clases.png": "Distribución de Clases — Variables Objetivo (J, K, L, M)",
        "02_comparacion_modelos.png": "Comparación de Modelos — F1 Ponderado (5-fold CV)",
        "03_curvas_aprendizaje.png": "Curvas de Aprendizaje — SVM Lineal + TF-IDF",
        "04_metricas_finales.png": "Métricas de Evaluación por Columna (Test Set)",
        "05_confusion_JKL.png": "Matrices de Confusión Normalizadas (J, K, L)",
        "06_confusion_M.png": "Matriz de Confusión Normalizada — Carrera (M)",
        "07_validacion_cruzada.png": "Distribución F1 Score — Validación Cruzada 5-fold"
    }

    # Desplegar los gráficos almacenados en GRAFICOS_DIR
    graficos_encontrados = False
    
    for filename, title in graficos_archivos.items():
        image_path = os.path.join(GRAFICOS_DIR, filename)
        if os.path.exists(image_path):
            graficos_encontrados = True
            st.markdown(f"#### 📈 {title}")
            st.image(image_path, use_container_width=True)
            st.divider()

    if not graficos_encontrados:
        st.warning(f"No se encontraron gráficos guardados en la ruta `{GRAFICOS_DIR}`. Ejecute la evaluación con `evaluador.py` para generarlos.")