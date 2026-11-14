"""
Fase 5 - Modelos con ajuste de hiperparámetros.

Compara tres modelos sobre las 56 características, junto a las referencias:
  - Ridge:          regresión lineal regularizada (modelo sencillo).
  - Random Forest:  muchos árboles de decisión promediados (no lineal).
  - MLP:            perceptrón multicapa, la red de neuronas del temario.

CÓMO SE AJUSTAN LOS HIPERPARÁMETROS SIN HACER TRAMPA (validación anidada)
Un hiperparámetro es un ajuste del modelo que no se aprende solo (p. ej. cuántas
neuronas tiene la red). Si se eligiera mirando el error en los datos de prueba,
ese error dejaría de ser honesto. Por eso hay dos niveles:
  - Nivel externo: las 5 particiones por jugadora de siempre. Miden el error.
  - Nivel interno: DENTRO de los datos de entrenamiento de cada partición se
    hace otra validación cruzada (3 particiones, también por jugadora) para
    elegir la mejor combinación. Los datos de prueba no intervienen.

Uso (desde la carpeta "Practica final"); tarda unos minutos:
    .venv/bin/python src/modelos.py
"""

import os
import time
import warnings

# Avisos falsos de numpy en macOS al multiplicar matrices (ver modelo_ridge.py).
# La variable de entorno hace que también se silencien en los procesos en paralelo.
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import TODAS
from validacion import (RAIZ, cargar_ejemplos, metricas, particion_temporal,
                        particiones_por_jugadora)

FIGURAS = RAIZ / "reports" / "figures"
TABLAS = RAIZ / "reports" / "tablas"
AZUL, NARANJA, VERDE, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
GRIS_CLARO, TINTA = "#e1e0d9", "#52514e"
SEMILLA = 0  # fija, para que los resultados se puedan reproducir

REFERENCIAS = {"Persistencia": "readiness", "Media de 7 días": "readiness_m7"}
COLORES = {"Ridge": AZUL, "Random Forest": NARANJA, "MLP": VERDE}


def definir_modelos() -> dict:
    """Cada modelo es un Pipeline (pasos encadenados) más la rejilla de
    hiperparámetros que se va a probar. El relleno de vacíos y el escalado van
    dentro del Pipeline: se ajustan solo con los datos de entrenamiento."""
    rellenar = ("rellenar", SimpleImputer(strategy="median"))
    escalar = ("escalar", StandardScaler())
    return {
        "Ridge": (
            Pipeline([rellenar, escalar, ("modelo", Ridge())]),
            # alpha: fuerza de la regularización (más alto = coeficientes más pequeños)
            {"modelo__alpha": [0.1, 1, 10, 100, 1000]},
        ),
        "Random Forest": (
            # Los árboles no necesitan escalado
            Pipeline([rellenar, ("modelo", RandomForestRegressor(
                n_estimators=300, random_state=SEMILLA, n_jobs=1))]),
            {   # mínimo de ejemplos por hoja: más alto = árboles más simples
                "modelo__min_samples_leaf": [5, 20, 50],
                # fracción de características que mira cada división
                "modelo__max_features": [0.33, 1.0],
            },
        ),
        "MLP": (
            # early_stopping: reserva un 10 % del entrenamiento y para cuando
            # deja de mejorar, para no sobreajustar
            Pipeline([rellenar, escalar, ("modelo", MLPRegressor(
                max_iter=300, early_stopping=True, n_iter_no_change=15,
                random_state=SEMILLA))]),
            {   # capas ocultas: una de 16 o de 32 neuronas, o dos de 64 y 32
                "modelo__hidden_layer_sizes": [(16,), (32,), (64, 32)],
                # alpha: regularización de los pesos. En una primera prueba con
                # [0.001, 0.1] se elegía casi siempre el valor más alto, así
                # que la rejilla se amplió hacia arriba.
                "modelo__alpha": [0.1, 1, 10],
                # tasa de aprendizaje inicial
                "modelo__learning_rate_init": [0.001, 0.01],
            },
        ),
    }


def ajustar(nombre: str, entrenamiento: pd.DataFrame) -> GridSearchCV:
    """Elige los hiperparámetros con validación cruzada interna por jugadora
    (minimizando el MAE) y reentrena con todo el entrenamiento."""
    pipeline, rejilla = definir_modelos()[nombre]
    busqueda = GridSearchCV(pipeline, rejilla, cv=GroupKFold(n_splits=3),
                            scoring="neg_mean_absolute_error", n_jobs=-1)
    busqueda.fit(entrenamiento[TODAS], entrenamiento["objetivo"],
                 groups=entrenamiento["jugadora"])
    return busqueda


def evaluar(ejemplos: pd.DataFrame, particiones):
    """Devuelve métricas por partición, predicciones de cada ejemplo de prueba
    y los hiperparámetros elegidos en cada partición."""
    filas, predicciones, elegidos = [], [], []
    for nombre_part, idx_entrenamiento, idx_prueba in particiones:
        entrenamiento, prueba = ejemplos.iloc[idx_entrenamiento], ejemplos.iloc[idx_prueba]
        real = prueba["objetivo"].to_numpy()
        pred = {ref: prueba[col].to_numpy() for ref, col in REFERENCIAS.items()}
        for nombre in definir_modelos():
            inicio = time.time()
            busqueda = ajustar(nombre, entrenamiento)
            pred[nombre] = np.clip(busqueda.predict(prueba[TODAS]), 1, 10)
            elegidos.append({"particion": nombre_part, "modelo": nombre,
                             "MAE_interno": -busqueda.best_score_,
                             **{k.replace("modelo__", ""): str(v) for k, v in busqueda.best_params_.items()}})
            print(f"  {nombre_part} · {nombre}: {time.time() - inicio:.0f} s", flush=True)
        for metodo, p in pred.items():
            filas.append({"particion": nombre_part, "metodo": metodo, **metricas(real, p)})
        predicciones.append(pd.DataFrame({
            "particion": nombre_part, "jugadora": prueba["jugadora"].to_numpy(),
            "fecha": prueba["fecha"].to_numpy(), "hoy": prueba["readiness"].to_numpy(),
            "real": real, **pred}))
    return pd.DataFrame(filas), pd.concat(predicciones, ignore_index=True), pd.DataFrame(elegidos)


def resumir(cv: pd.DataFrame, orden: list[str]) -> pd.DataFrame:
    r = cv.groupby("metodo")[["MAE", "RMSE", "R2"]].agg(["mean", "std"])
    r.columns = [f"{m}_{'media' if e == 'mean' else 'desv'}" for m, e in r.columns]
    return r.loc[orden]


def errores_por_jugadora(pred: pd.DataFrame, metodos: list[str]) -> pd.DataFrame:
    """MAE de cada método para cada jugadora."""
    err = pred[metodos].sub(pred["real"], axis=0).abs()
    return err.groupby(pred["jugadora"]).mean()


def pruebas_estadisticas(cv: pd.DataFrame, pred: pd.DataFrame, modelos: list[str]) -> pd.DataFrame:
    """Comparaciones por pares: t-test pareado sobre el MAE de las 5 particiones
    y Wilcoxon pareado sobre el MAE de las 50 jugadoras. p < 0,05 = significativa."""
    por_part = cv.pivot(index="particion", columns="metodo", values="MAE")
    por_jug = errores_por_jugadora(pred, list(REFERENCIAS) + modelos)
    pares = [(m, "Media de 7 días") for m in modelos] + \
            [(a, b) for i, a in enumerate(modelos) for b in modelos[i + 1:]]
    filas = []
    for a, b in pares:
        filas.append({"comparacion": f"{a}  vs  {b}",
                      "dif_MAE": por_part[a].mean() - por_part[b].mean(),
                      "p_ttest_particiones": stats.ttest_rel(por_part[a], por_part[b]).pvalue,
                      "p_wilcoxon_jugadoras": stats.wilcoxon(por_jug[a], por_jug[b]).pvalue,
                      "jugadoras_en_que_gana_el_primero": int((por_jug[a] < por_jug[b]).sum())})
    return pd.DataFrame(filas).set_index("comparacion")


def anova(cv: pd.DataFrame, pred: pd.DataFrame, modelos: list[str]) -> dict:
    """¿Hay diferencias entre los tres modelos en conjunto?
    - ANOVA de un factor sobre el MAE de las 5 particiones (como el operador
      ANOVA de AI Studio).
    - Friedman sobre el MAE por jugadora: la versión que no supone normalidad
      y tiene en cuenta que son las mismas jugadoras en los tres modelos."""
    por_part = cv.pivot(index="particion", columns="metodo", values="MAE")
    por_jug = errores_por_jugadora(pred, modelos)
    return {"p_anova_particiones": stats.f_oneway(*[por_part[m] for m in modelos]).pvalue,
            "p_friedman_jugadoras": stats.friedmanchisquare(*[por_jug[m] for m in modelos]).pvalue}


def desglose(pred: pd.DataFrame, metodos: list[str]) -> pd.DataFrame:
    """MAE según si la readiness cambia y según su nivel real."""
    err = pred[metodos].sub(pred["real"], axis=0).abs()
    tipo = np.where(pred["real"] == pred["hoy"], "Mañana = hoy", "Mañana ≠ hoy")
    nivel = pd.cut(pred["real"], [0, 4, 7, 10], labels=["Baja (1-4)", "Media (5-7)", "Alta (8-10)"])
    return pd.concat([err.groupby(tipo).mean().T, err.groupby(nivel, observed=True).mean().T], axis=1)


def estilo() -> None:
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": TINTA,
        "xtick.color": GRIS, "ytick.color": TINTA, "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True, "legend.frameon": False})


def fig_comparacion(resumen: pd.DataFrame) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    colores = [GRIS if m in REFERENCIAS else AZUL for m in resumen.index]
    for ax, metrica in zip(ejes, ("MAE", "RMSE")):
        media, desv = resumen[f"{metrica}_media"], resumen[f"{metrica}_desv"]
        barras = ax.barh(resumen.index, media, xerr=desv, color=colores, height=0.6,
                         error_kw={"ecolor": TINTA, "elinewidth": 1.2, "capsize": 3})
        for barra, m, s in zip(barras, media, desv):
            ax.text(m + s + 0.02, barra.get_y() + barra.get_height() / 2,
                    f"{m:.3f}".replace(".", ","), va="center", fontsize=9, color=TINTA)
        ax.set_xlim(0, 1.6)
        ax.set_xlabel(f"{metrica} (menos es mejor)")
        ax.set_title(metrica)
        ax.grid(axis="x", color=GRIS_CLARO, linewidth=0.8)
        ax.invert_yaxis()
    ejes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GRIS), plt.Rectangle((0, 0), 1, 1, color=AZUL)],
                   labels=["Referencia", "Modelo ajustado"], loc="lower right", fontsize=8)
    fig.suptitle("Modelos frente a las referencias, validación por jugadoras (media de 5 particiones ± desv. típica)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURAS / "13_modelos_vs_baselines.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_prediccion_por_nivel(pred: pd.DataFrame, modelos: list[str]) -> None:
    """Para cada valor real de readiness, qué predice de media cada modelo.
    Un modelo perfecto seguiría la diagonal."""
    media = pred.groupby("real")[modelos].mean()
    n = pred.groupby("real").size()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot([1, 10], [1, 10], color="#c3c2b7", linewidth=1, linestyle="--")
    ax.annotate("Predicción perfecta", (2.2, 2.2), xytext=(3.0, 1.6), color=TINTA, fontsize=8,
                arrowprops={"arrowstyle": "-", "color": "#c3c2b7"})
    for m in modelos:
        ax.plot(media.index, media[m], color=COLORES[m], linewidth=2, marker="o", markersize=5, label=m)
    for x, k in n.items():
        ax.text(x, 9.75, f"{k}", ha="center", fontsize=7, color=GRIS)
    ax.text(0.6, 9.75, "n:", ha="right", fontsize=7, color=GRIS)
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(1, 10)
    ax.set_xlabel("Readiness real de mañana")
    ax.set_ylabel("Predicción media")
    ax.set_title("Los modelos se quedan cortos en los extremos")
    ax.grid(color=GRIS_CLARO, linewidth=0.8)
    ax.legend(loc="lower right")
    fig.savefig(FIGURAS / "14_prediccion_por_nivel.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    TABLAS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220, "display.float_format", "{:.4f}".format)
    estilo()
    ejemplos = cargar_ejemplos()
    modelos = list(definir_modelos())
    orden = list(REFERENCIAS) + modelos

    print("=== VALIDACIÓN POR JUGADORAS (5 particiones, ajuste interno de 3) ===")
    cv, pred, elegidos = evaluar(ejemplos, particiones_por_jugadora(ejemplos))
    resumen = resumir(cv, orden)
    print(resumen.to_string())
    print("\nMAE por partición:")
    print(cv.pivot(index="particion", columns="metodo", values="MAE")[orden].to_string())
    print("\nHiperparámetros elegidos en cada partición:")
    print(elegidos.fillna("").to_string(index=False))
    pruebas = pruebas_estadisticas(cv, pred, modelos)
    print("\nComparaciones por pares (dif_MAE negativa = gana el primero):")
    print(pruebas.to_string())
    conjunto = anova(cv, pred, modelos)
    print("\nDiferencias entre los tres modelos en conjunto:", {k: f"{v:.2g}" for k, v in conjunto.items()})
    tabla_desglose = desglose(pred, orden)
    print("\nMAE según el tipo de día:")
    print(tabla_desglose.to_string())

    print("\n=== COMPROBACIÓN TEMPORAL (entrenar 2020, evaluar 2021) ===")
    temporal, _, elegidos_t = evaluar(ejemplos, particion_temporal(ejemplos))
    temporal = temporal.set_index("metodo").loc[orden, ["MAE", "RMSE", "R2"]]
    print(temporal.to_string())

    resumen.to_csv(TABLAS / "modelos_resumen.csv", float_format="%.4f")
    cv.to_csv(TABLAS / "modelos_por_particion.csv", index=False, float_format="%.4f")
    elegidos.to_csv(TABLAS / "modelos_hiperparametros.csv", index=False, float_format="%.4f")
    pruebas.to_csv(TABLAS / "modelos_pruebas_estadisticas.csv", float_format="%.5f")
    pd.Series(conjunto).to_csv(TABLAS / "modelos_anova.csv", header=["p_valor"], float_format="%.5f")
    tabla_desglose.to_csv(TABLAS / "modelos_por_tipo_de_dia.csv", float_format="%.4f")
    temporal.to_csv(TABLAS / "modelos_temporal.csv", float_format="%.4f")
    elegidos_t.to_csv(TABLAS / "modelos_hiperparametros_temporal.csv", index=False, float_format="%.4f")
    # Predicción de cada ejemplo cuando estaba en la partición de prueba: sirve
    # para el análisis de errores posterior sin tener que reentrenar.
    pred.to_csv(TABLAS / "predicciones_cv.csv", index=False, float_format="%.3f", date_format="%Y-%m-%d")
    fig_comparacion(resumen)
    fig_prediccion_por_nivel(pred, modelos)


if __name__ == "__main__":
    main()
