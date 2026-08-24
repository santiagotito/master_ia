
import os
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import GRAFICOS_DIR, TARGET_COLS, TARGET_LABELS, SEED


os.makedirs(GRAFICOS_DIR, exist_ok=True) # verifica si el directorio existe, si no, lo crea

sns.set_style('whitegrid')


def _obtener_nombre_columna(df, target_key, default_name):
    """Auxiliar para obtener el nombre real de la columna en df o usar un fallback."""
    col_config = TARGET_COLS.get(target_key, default_name)
    if col_config in df.columns:
        return col_config
    if default_name in df.columns:
        return default_name
    # Búsqueda por subcadena
    for col in df.columns:
        if default_name.lower() in col.lower() or target_key.lower() == col.lower():
            return col
    return df.columns[0] if len(df.columns) > 0 else default_name


def make_tfidf():
    """Genera el vectorizador TF-IDF con la configuración base."""
    return TfidfVectorizer(
        ngram_range=(1, 2), max_features=10000,
        sublinear_tf=True, strip_accents='unicode', min_df=2
    )


def guardar_distribucion_clases(df):
    """Genera y guarda el gráfico 01_distribucion_clases.png."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('Distribución de Clases — Variables Objetivo (J, K, L, M)', fontsize=14, fontweight='bold')

    palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
               '#1abc9c', '#e67e22', '#34495e', '#c0392b', '#16a085',
               '#8e44ad', '#2980b9', '#27ae60']

    # Identificar nombres reales de las columnas
    col_j = _obtener_nombre_columna(df, 'J', 'tipo_campania')
    col_k = _obtener_nombre_columna(df, 'K', 'modalidad')
    col_l = _obtener_nombre_columna(df, 'L', 'grado')
    col_m = _obtener_nombre_columna(df, 'M', 'carrera')

    # --- J: Tipo de Campaña ---
    ax = axes[0, 0]
    vc = df[col_j].value_counts()
    bars = ax.bar(vc.index, vc.values, color=palette[:len(vc)], edgecolor='white')
    ax.set_title(f'J — Tipo de Campaña ({len(vc)} clases)', fontsize=11)
    ax.set_ylabel('Frecuencia')
    for bar, v in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                f'{v:,}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.4)
    ax.set_ylim([0, vc.max() * 1.2 if not vc.empty else 10])

    # --- K: Modalidad ---
    ax = axes[0, 1]
    vc = df[col_k].value_counts()
    bars = ax.barh(vc.index, vc.values, color=palette[:len(vc)], edgecolor='white')
    ax.set_title(f'K — Modalidad ({len(vc)} clases)', fontsize=11)
    ax.set_xlabel('Frecuencia')
    for bar, v in zip(bars, vc.values):
        ax.text(v + 8, bar.get_y() + bar.get_height()/2, f'{v:,}', va='center', fontsize=8)
    ax.grid(axis='x', alpha=0.4)

    # --- L: Grado ---
    ax = axes[1, 0]
    vc = df[col_l].value_counts()
    colors_l = plt.cm.tab20(np.linspace(0, 1, len(vc))) if len(vc) > 0 else 'blue'
    bars = ax.barh(vc.index, vc.values, color=colors_l, edgecolor='white')
    ax.set_title(f'L — Grado ({len(vc)} clases)', fontsize=11)
    ax.set_xlabel('Frecuencia')
    for bar, v in zip(bars, vc.values):
        ax.text(v + 3, bar.get_y() + bar.get_height()/2, f'{v:,}', va='center', fontsize=7)
    ax.grid(axis='x', alpha=0.4)

    # --- M: Carrera (top 20) ---
    ax = axes[1, 1]
    vc = df[col_m].value_counts().head(20)
    colors_m = plt.cm.viridis(np.linspace(0.15, 0.85, len(vc))) if len(vc) > 0 else 'viridis'
    bars = ax.barh(vc.index, vc.values, color=colors_m, edgecolor='white')
    ax.set_title(f'M — Carrera (top {len(vc)} de {df[col_m].nunique()} clases)', fontsize=11)
    ax.set_xlabel('Frecuencia')
    for bar, v in zip(bars, vc.values):
        ax.text(v + 1, bar.get_y() + bar.get_height()/2, f'{v}', va='center', fontsize=7)
    ax.grid(axis='x', alpha=0.4)

    plt.tight_layout()
    output_path = os.path.join(GRAFICOS_DIR, '01_distribucion_clases.png')
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"✅ Figura guardada en: {output_path}")

    print("\n=== RATIO DE DESBALANCE (frecuencia_max / frecuencia_min) ===")
    for cn, col in TARGET_COLS.items():
        if col in df.columns and not df[col].empty:
            vc_col = df[col].value_counts()
            ratio = vc_col.iloc[0] / (vc_col.iloc[-1] + 1e-9)
            print(f"  {cn} ({col}): {ratio:6.1f}× | max={vc_col.iloc[0]:4d} ({vc_col.index[0]}) | min={vc_col.iloc[-1]:4d} ({vc_col.index[-1]})")


def guardar_comparacion_modelos(cv_results, model_names=None):
    """Genera y guarda el gráfico 02_comparacion_modelos.png."""
    if not cv_results:
        print("⚠️ cv_results está vacío — no se puede generar el gráfico 02.")
        return

    if model_names is None:
        first_target = list(cv_results.keys())[0]
        model_names = list(cv_results[first_target].keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Comparación de Modelos — F1 Ponderado (5-fold CV)', fontsize=14, fontweight='bold')

    colors_m = ['#95a5a6', '#3498db', '#2ecc71', '#e74c3c']

    for ax, (cn, col) in zip(axes.flatten(), TARGET_COLS.items()):
        if cn not in cv_results:
            continue
        means = [cv_results[cn][m].mean() for m in model_names]
        stds  = [cv_results[cn][m].std()  for m in model_names]
        bars  = ax.bar(range(len(model_names)), means, yerr=stds,
                       capsize=5, color=colors_m[:len(model_names)], edgecolor='white', linewidth=0.8)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=20, ha='right', fontsize=8)
        ax.set_title(TARGET_LABELS.get(cn, col), fontsize=11)
        ax.set_ylabel('F1 Score Ponderado')
        ax.set_ylim([0, 1.15])
        ax.axhline(means[0], color='gray', linestyle=':', linewidth=1, alpha=0.6, label='Baseline')
        ax.grid(axis='y', alpha=0.3)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.01,
                    f'{m:.3f}', ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(GRAFICOS_DIR, '02_comparacion_modelos.png')
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"✅ Figura guardada en: {output_path}")


def guardar_curvas_aprendizaje(df_clean):
    """Genera y guarda el gráfico 03_curvas_aprendizaje.png calculando la curva de aprendizaje."""
    X_all = np.array(df_clean["texto"].tolist())
    train_sizes_pct = np.linspace(0.1, 1.0, 8)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Curvas de Aprendizaje — SVM Lineal + TF-IDF', fontsize=14, fontweight='bold')

    for ax, (cn, col) in zip(axes.flatten(), TARGET_COLS.items()):
        y_col = df_clean[col].to_numpy(dtype=str)

        pipe_lc = Pipeline([
            ('tfidf', make_tfidf()),
            ('clf',   LinearSVC(C=1, max_iter=2000, class_weight='balanced', random_state=SEED))
        ])

        sizes, tr_sc, val_sc = learning_curve(
            pipe_lc, X_all, y_col,
            train_sizes=train_sizes_pct, cv=5,
            scoring='f1_weighted', n_jobs=1,
            random_state=SEED
        )

        tr_mean, tr_std   = tr_sc.mean(1), tr_sc.std(1)
        val_mean, val_std = val_sc.mean(1), val_sc.std(1)

        ax.plot(sizes, tr_mean,  'o-',  color='steelblue', label='Entrenamiento')
        ax.fill_between(sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color='steelblue')
        ax.plot(sizes, val_mean, 's--', color='tomato',    label='Validación')
        ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='tomato')

        gap = tr_mean[-1] - val_mean[-1]
        ax.set_title(f"{TARGET_LABELS[cn]}  (gap={gap:.3f})", fontsize=10)
        ax.set_xlabel('Muestras de entrenamiento')
        ax.set_ylabel('F1 Ponderado')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

    plt.tight_layout()
    output_path = os.path.join(GRAFICOS_DIR, '03_curvas_aprendizaje.png')
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"✅ Figura guardada en: {output_path}")


def _extraer_y_true(y_tests, cn, col, df_clean=None):
    """Extrae las etiquetas reales soportando diccionarios, DataFrames o fallback desde df_clean."""
    if isinstance(y_tests, dict):
        if cn in y_tests:
            return y_tests[cn]
        if col in y_tests:
            return y_tests[col]
    elif isinstance(y_tests, pd.DataFrame):
        if col in y_tests.columns:
            return y_tests[col].values
        if cn in y_tests.columns:
            return y_tests[cn].values

    if df_clean is not None:
        if col in df_clean.columns:
            return df_clean[col].values
        if cn in df_clean.columns:
            return df_clean[cn].values

    raise KeyError(f"No se pudo extraer la etiqueta real para '{cn}' ({col}).")


def guardar_metricas_finales(final_models, X_test, y_tests):
    """Calcula las métricas de evaluación en el test set y guarda el gráfico 04_metricas_finales.png."""
    metrics_data = {cn: {} for cn in TARGET_COLS}
    for cn, col in TARGET_COLS.items():
        y_pred = final_models[cn].predict(X_test)
        y_true = _extraer_y_true(y_tests, cn, col)
        metrics_data[cn] = {
            'Accuracy':    accuracy_score(y_true, y_pred),
            'Precisión':   precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall':      recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1 Macro':    f1_score(y_true, y_pred, average='macro', zero_division=0),
            'F1 Weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }

    df_metrics = pd.DataFrame(metrics_data).T
    print("=== TABLA DE MÉTRICAS (conjunto de test) ===")
    print(df_metrics.round(4).to_string())

    metric_names = list(df_metrics.columns)
    x = np.arange(len(metric_names))
    width = 0.18
    col_colors_bar = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (cn, col) in enumerate(TARGET_COLS.items()):
        vals = [metrics_data[cn][m] for m in metric_names]
        bars = ax.bar(x + i*width, vals, width, label=TARGET_LABELS[cn],
                      color=col_colors_bar[i], alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=6.5, rotation=90)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylabel('Score')
    ax.set_title('Métricas de Evaluación por Columna — Conjunto de Test', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim([0, 1.18])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(GRAFICOS_DIR, '04_metricas_finales.png')
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"✅ Figura guardada en: {output_path}")

    return df_metrics


def imprimir_reporte_clasificacion(final_models, X_test, y_tests):
    """Imprime el reporte detallado de clasificación para cada variable objetivo."""
    for cn, col in TARGET_COLS.items():
        y_pred = final_models[cn].predict(X_test)
        y_true = _extraer_y_true(y_tests, cn, col)
        print(f"\n{'═'*60}")
        print(f" {TARGET_LABELS[cn].upper()}")
        print(f"{'═'*60}")

        if cn == 'M':
            rep = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
            print(f"  Accuracy:         {rep['accuracy']:.4f}")
            print(f"  F1 Macro:         {rep['macro avg']['f1-score']:.4f}")
            print(f"  F1 Weighted:      {rep['weighted avg']['f1-score']:.4f}")
            print(f"  Clases presentes en test: {len(set(y_true))}")
            print(f"  Clases predichas:         {len(set(y_pred))}")
            print(f"\n  Top 20 clases por soporte:")
            cdf = pd.DataFrame(rep).T
            cdf = cdf[~cdf.index.isin(['accuracy', 'macro avg', 'weighted avg'])]
            cdf = cdf.sort_values('support', ascending=False).head(20)
            cdf[['precision', 'recall', 'f1-score', 'support']] = cdf[['precision', 'recall', 'f1-score', 'support']].apply(pd.to_numeric)
            print(cdf[['precision', 'recall', 'f1-score', 'support']].round(3).to_string())
        else:
            print(classification_report(y_true, y_pred, zero_division=0))


def analizar_errores_frecuentes(final_models, X_test, y_tests):
    """Muestra las confusiones más frecuentes (Real -> Predicho) y ejemplos de piezas mal clasificadas."""
    X_test_arr = np.array(X_test)
    
    for cn, col in list(TARGET_COLS.items())[:3]:   # Solo J, K, L
        y_pred = final_models[cn].predict(X_test)
        y_real = np.array(_extraer_y_true(y_tests, cn, col))
        err_idx = np.where(y_pred != y_real)[0]

        print(f"\n{'─'*55}")
        print(f" {TARGET_LABELS[cn].upper()}")
        print(f" Errores: {len(err_idx)} / {len(y_real)} ({len(err_idx)/len(y_real)*100:.1f}%)")
        print(f"{'─'*55}")

        pairs = Counter([(y_real[i], y_pred[i]) for i in err_idx])
        print("  Confusiones más frecuentes (Real → Predicho | n):")
        for (real, pred), cnt in pairs.most_common(6):
            print(f"    {real:<25} → {pred:<25} | {cnt}×")

        print("  Ejemplos de piezas mal clasificadas:")
        for i in err_idx[:4]:
            texto_snippet = str(X_test_arr[i])[:65].replace('\n', ' ')
            print(f"    '{texto_snippet}'")
            print(f"       Real: {y_real[i]:<25} | Predicho: {y_pred[i]}")


def guardar_matrices_confusion_jkl(models, X_test, y_tests, df_clean):
    """Genera y guarda la matriz de confusión normalizada para J, K y L."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Matrices de Confusión Normalizadas — J, K, L', fontsize=13, fontweight='bold')

    for ax, (cn, col) in zip(axes, list(TARGET_COLS.items())[:3]):
        y_pred = models[cn].predict(X_test)
        y_true = _extraer_y_true(y_tests, cn, col, df_clean)
        
        # Obtener lista de etiquetas unificadas
        col_df_name = col if col in df_clean.columns else _obtener_nombre_columna(df_clean, cn, col)
        labels = sorted(df_clean[col_df_name].unique()) if col_df_name in df_clean.columns else sorted(list(set(y_true) | set(y_pred)))
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

        sns.heatmap(
            cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            ax=ax, linewidths=0.4, vmin=0, vmax=1,
            annot_kws={'size': 8}
        )
        ax.set_title(TARGET_LABELS[cn], fontsize=10)
        ax.set_xlabel('Predicho')
        ax.set_ylabel('Real')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)

    plt.tight_layout()
    output_path = os.path.join(GRAFICOS_DIR, '05_confusion_JKL.png')
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"✅ Figura guardada en: {output_path}")


def guardar_matriz_confusion_m(final_models, X_test, y_tests, df_clean, top_n=25):
    """Genera y guarda la matriz de confusión normalizada para la columna M (carrera) en las Top N clases."""
    col_m = _obtener_nombre_columna(df_clean, 'M', 'carrera')
    top_classes = df_clean[col_m].value_counts().head(top_n).index.tolist()

    y_real_m = np.array(_extraer_y_true(y_tests, 'M', col_m, df_clean))
    y_pred_m = final_models['M'].predict(X_test)

    mask = np.isin(y_real_m, top_classes)
    y_r = y_real_m[mask]
    y_p = np.where(np.isin(y_pred_m[mask], top_classes), y_pred_m[mask], 'OTRA')

    labels_plot = top_classes
    cm_m = confusion_matrix(y_r, y_p, labels=labels_plot)
    cm_m_norm = cm_m.astype(float) / (cm_m.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(17, 15))
    sns.heatmap(
        cm_m_norm, cmap='Blues', linewidths=0.2,
        xticklabels=labels_plot, yticklabels=labels_plot,
        ax=ax, vmin=0, vmax=1
    )
    for i in range(len(labels_plot)):
        for j in range(len(labels_plot)):
            v = cm_m_norm[i, j]
            if v >= 0.05:
                ax.text(j + 0.5, i + 0.5, f'{v:.2f}',
                        ha='center', va='center', fontsize=6,
                        color='white' if v > 0.55 else 'black')

    ax.set_title(f'M — Carrera: Matriz de Confusión Normalizada (Top {top_n} clases)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicho', fontsize=11)
    ax.set_ylabel('Real', fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    plt.tight_layout()

    output_path = os.path.join(GRAFICOS_DIR, '06_confusion_M.png')
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"✅ Figura guardada en: {output_path}")

    diag_acc = np.diag(cm_m_norm)
    class_acc_df = pd.DataFrame({'Carrera': labels_plot, 'Accuracy': diag_acc})
    class_acc_df = class_acc_df.sort_values('Accuracy', ascending=False)
    print(f"\nAccuracy por clase (Top {top_n}):")
    print(class_acc_df.to_string(index=False))

    return class_acc_df


def guardar_validacion_cruzada_final(final_models, df_clean, cv_kfold=5):
    """Genera y guarda el gráfico 07_validacion_cruzada.png."""
    col_colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
    X_all = np.array(df_clean["texto"].tolist())

    cv_final = {}
    for cn, col in TARGET_COLS.items():
        pipe = final_models[cn]
        y_col = df_clean[col].to_numpy(dtype=str)
        scores = cross_val_score(pipe, X_all, y_col, cv=cv_kfold, scoring='f1_weighted', n_jobs=1)
        cv_final[cn] = scores

    fig, ax = plt.subplots(figsize=(8, 4))
    bp = ax.boxplot([cv_final[cn] for cn in TARGET_COLS], patch_artist=True,
                    labels=[f"{cn}\n{col}" for cn, col in TARGET_COLS.items()],
                    medianprops=dict(color='black', linewidth=2))

    for patch, color in zip(bp['boxes'], col_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title('Distribución F1 Score — Validación Cruzada 5-fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Ponderado')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(GRAFICOS_DIR, '07_validacion_cruzada.png')
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"✅ Figura guardada en: {output_path}")
    return cv_final
