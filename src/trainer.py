"""
Entrena pipelines independientes (TF-IDF + LinearSVC Calibrado) para cada columna objetivo y guarda el objeto final usando      
"""
import joblib
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config import TARGET_COLS, SEED, MODEL_PATH, EXCEL_PATH
from src.data_loader import load_and_preprocess_data
from src.evaluador import (guardar_distribucion_clases,
    guardar_comparacion_modelos,
    guardar_curvas_aprendizaje,
    guardar_metricas_finales,
    guardar_matrices_confusion_jkl,
    guardar_matriz_confusion_m,
    guardar_validacion_cruzada_final,
    ) # 👈 Importamos la función


# Define la ruta del directorio para los gráficos
GRAFICOS_DIR = os.path.join("src", "graficos")  # O la ruta que prefieras, ej: "graficos"

def make_tfidf():
    """Genera el vectorizador TF-IDF configurado."""
    return TfidfVectorizer(
        ngram_range=(1, 2), max_features=10000,
        sublinear_tf=True, strip_accents='unicode', min_df=2
    )


def  entrenar_modelos(df_clean):    ##comparar_modelos_cv(df_clean):
    """Ejecuta Validación Cruzada Estratificada para comparar clasificadores."""
    X_all = np.array(df_clean["texto"].tolist())
    
    # LinearSVC directo (sin CalibratedClassifierCV) para la comparación
    # — CalibratedClassifierCV(cv=3) falla cuando hay clases con < 3 ejemplos
    MODELS = {
        'Dummy':          DummyClassifier(strategy='most_frequent', random_state=SEED),
        'Naive Bayes':    ComplementNB(alpha=0.1),
        'Log. Regresión': LogisticRegression(C=5, max_iter=2000, class_weight='balanced',
                                             solver='saga', random_state=SEED),
        'SVM Lineal':     LinearSVC(C=1, max_iter=2000, class_weight='balanced', random_state=SEED),
    }

    cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_results = {}

    print("\n=== COMPARACIÓN DE MODELOS (5-Fold Cross Validation) ===")
    print(f"{'Col':<6} {'Modelo':<20} {'F1 Media':>10} {'± Std':>8}")
    print("-" * 48)

    for cn, col in TARGET_COLS.items():
        y_col = np.array(df_clean[col].tolist())
        cv_results[cn] = {}
        for mname, clf in MODELS.items():
            pipe = Pipeline([('tfidf', make_tfidf()), ('clf', clf)])
            scores = cross_val_score(pipe, X_all, y_col, cv=cv_kfold,
                                     scoring='f1_weighted', n_jobs=1)
            cv_results[cn][mname] = scores
            print(f"  {cn:<4} {mname:<20} {scores.mean():>10.4f} ±{scores.std():>6.4f}")
        print()

    return cv_results


def train_and_save_models(excel_path, output_model_path=MODEL_PATH):
    df_clean = load_and_preprocess_data(excel_path)
    X_all = np.array(df_clean["texto"].tolist())
    
    # Parámetros óptimos identificados previamente por GridSearch
    best_params = {
        'J': {'ngram': (1, 2), 'max_feat': 10000, 'C': 1.0},
        'K': {'ngram': (1, 2), 'max_feat': 10000, 'C': 1.0},
        'L': {'ngram': (1, 2), 'max_feat': 10000, 'C': 1.0},
        'M': {'ngram': (1, 2), 'max_feat': 20000, 'C': 5.0},
    }

    models = {}
    metrics = {}
    soporte_clases = {}

    for cn, col in TARGET_COLS.items():
        bp = best_params.get(cn, {'ngram': (1, 2), 'max_feat': 10000, 'C': 1.0})
        y_col = np.array(df_clean[col].tolist())

        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=bp['ngram'],
                max_features=bp['max_feat'],
                sublinear_tf=True, strip_accents='unicode', min_df=2
            )),
            ('clf', CalibratedClassifierCV(
                LinearSVC(C=bp['C'], max_iter=3000, class_weight='balanced', random_state=SEED),
                cv=None
            ))
        ])

        pipe.fit(X_all, y_col)
        models[cn] = pipe

        clases, cuentas = np.unique(y_col, return_counts=True)
        soporte_clases[col] = dict(zip(clases.tolist(), cuentas.tolist()))

        y_pred = pipe.predict(X_all)
        metrics[cn] = {
            'Accuracy': accuracy_score(y_col, y_pred),
            'F1 Weighted': f1_score(y_col, y_pred, average='weighted', zero_division=0)
        }

        # Guardar modelo final (.pkl) -- incluye soporte_clases (Story: selector de
        # modelos guardados) para que un .pkl cargado después pueda aplicar el mismo
        # umbral de confianza + soporte mínimo que el modelo entrenado en vivo.
    joblib.dump({"pipelines": models, "soporte_clases": soporte_clases}, output_model_path)
    print(f"✅ Modelos guardados en: {output_model_path}")
    
    # 📊 Guardar matrices de confusión y gráficos de diagnóstico
    try:
        # Construimos el diccionario de etiquetas reales para cada variable objetivo
        y_targets = {cn: np.array(df_clean[col].tolist()) for cn, col in TARGET_COLS.items()}
        guardar_matrices_confusion_jkl(models, X_all, y_targets, df_clean)
    except Exception as e:
        print(f"Error generando gráficos: {e}")

    return metrics

def preparar_datos_y_entrenar(excel_path):
    """Carga datos, aplica la división Train/Test y entrena modelos finales."""
    df_clean = load_and_preprocess_data(excel_path)
    X_all = np.array(df_clean["texto"].tolist())

    # Split 80/20 idéntico al notebook original
    idx = np.arange(len(df_clean))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=SEED,
        stratify=df_clean["tipo_campania"].tolist()
    )
    
    X_train, X_test = X_all[idx_train], X_all[idx_test]
    y_trains = {cn: np.array(df_clean[col].tolist())[idx_train] for cn, col in TARGET_COLS.items()}
    y_tests = {cn: np.array(df_clean[col].tolist())[idx_test] for cn, col in TARGET_COLS.items()}

    best_params = {
        'J': {'ngram': (1, 2), 'max_feat': 10000, 'C': 1.0},
        'K': {'ngram': (1, 2), 'max_feat': 10000, 'C': 1.0},
        'L': {'ngram': (1, 2), 'max_feat': 10000, 'C': 1.0},
        'M': {'ngram': (1, 2), 'max_feat': 20000, 'C': 5.0},
    }

    final_models = {}
    for cn, col in TARGET_COLS.items():
        bp = best_params[cn]
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=bp['ngram'], max_features=bp['max_feat'],
                sublinear_tf=True, strip_accents='unicode', min_df=2
            )),
            ('clf', CalibratedClassifierCV(
                LinearSVC(C=bp['C'], max_iter=3000, class_weight='balanced', random_state=SEED),
                cv=None
            ))
        ])
        pipe.fit(X_train, y_trains[cn])
        final_models[cn] = pipe

    return df_clean, final_models, X_test, y_tests

    # ==============================================================================
# PUNTO DE ENTRADA PRINCIPAL PARA STREAMLIT / CONSOLA
# ==============================================================================
def main(excel_path=None, output_model_path=None):
    """Inicia el proceso completo de entrenamiento. `excel_path` permite
    entrenar con un Excel distinto al configurado por defecto (ej. uno
    subido a mano desde la app) -- si no se pasa, usa EXCEL_PATH.
    `output_model_path` permite guardar el .pkl resultante con otro nombre
    (ej. un modelo guardado aparte, seleccionable luego en la app) -- si no
    se pasa, pisa el modelo por defecto (MODEL_PATH)."""
    excel_path = excel_path or EXCEL_PATH
    output_model_path = output_model_path or MODEL_PATH
    print("Iniciando entrenamiento de modelos...")
    metricas = train_and_save_models(excel_path, output_model_path)

    # Asegurar que el directorio de gráficos exista
    os.makedirs(GRAFICOS_DIR, exist_ok=True)

    # 1. Cargar y limpiar el DataFrame (asigna el resultado a df_clean)
    # Reemplaza 'cargar_datos()' por la función real que usas en tu proyecto
    # df_clean = load_and_preprocess_data()
    df_clean, final_models, X_test, y_tests = preparar_datos_y_entrenar(excel_path)

    # 2. Ejecutar entrenamiento y división de datos
    # Reemplaza esta línea por tu lógica real para obtener los modelos y sets de prueba
    # final_models, X_test, y_tests, cv_results = entrenar_modelos(df_clean)
    # 2. Ejecutar validación cruzada para la comparación de modelos (Devuelve 1 diccionario: cv_results)
    cv_results = entrenar_modelos(df_clean)

    # --------------------------------------------------------------------------
    # GENERACIÓN Y GUARDADO DE GRÁFICOS DE EVALUACIÓN
    # --------------------------------------------------------------------------
    print("\n📊 --- GENERANDO Y GUARDANDO GRÁFICOS DE EVALUACIÓN ---")

    # Gráfico 01
    try:
        guardar_distribucion_clases(df_clean)
    except Exception as e:
        print(f"❌ Error en gráfico 01: {e}")

    # Gráfico 02 (Requiere diccionario cv_results acumulado en validación)
    if 'cv_results' in locals() or 'cv_results' in globals():
        try:
            guardar_comparacion_modelos(cv_results)
        except Exception as e:
            print(f"❌ Error en gráfico 02: {e}")

    # Gráfico 03
    try:
        guardar_curvas_aprendizaje(df_clean)
    except Exception as e:
        print(f"❌ Error en gráfico 03: {e}")

    # Gráfico 04
    try:
        guardar_metricas_finales(final_models, X_test, y_tests)
    except Exception as e:
        print(f"❌ Error en gráfico 04: {e}")

    # Gráfico 05
    try:
        guardar_matrices_confusion_jkl(final_models, X_test, y_tests, df_clean)
    except Exception as e:
        print(f"❌ Error en gráfico 05: {e}")

    # Gráfico 06
    try:
        guardar_matriz_confusion_m(final_models, X_test, y_tests, df_clean)
    except Exception as e:
        print(f"❌ Error en gráfico 06: {e}")

    # Gráfico 07
    try:
        guardar_validacion_cruzada_final(final_models, df_clean)
    except Exception as e:
        print(f"❌ Error en gráfico 07: {e}")

    # Guardar artefactos finales
    # joblib.dump(final_models, MODEL_PATH)
    # print(f"\n✅ Proceso completado. Modelos guardados en: {MODEL_PATH}")
    
    print("✅ Proceso de generación de gráficos finalizado.")

    print("¡Entrenamiento finalizado con éxito!")
    print(metricas)

if __name__ == "__main__":
    main()