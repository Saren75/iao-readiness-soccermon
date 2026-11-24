"""
Fase 5 (primer modelo) - Regresión lineal Ridge.

Es el modelo más sencillo que tiene sentido probar: una suma ponderada de las
características. Sirve para dos cosas antes de pasar a modelos más complejos:
  1. Saber si aprender de los datos mejora al mejor baseline (media de 7 días).
  2. Saber qué bloque de características aporta: se entrena tres veces,
     añadiendo cada vez un bloque (ablación):
        A = historial de readiness + contexto
        B = A + resto del bienestar
        C = B + carga de entrenamiento

Sin fuga de información: el relleno de valores vacíos (mediana) y el escalado
se ajustan SOLO con los datos de entrenamiento de cada partición. Eso lo
garantiza el Pipeline de scikit-learn.

Todavía no se ajustan hiperparámetros (alpha = 1, el valor por defecto).

Uso (desde la carpeta "Practica final"):
    .venv/bin/python comprobacion/modelo_ridge.py
"""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import G_BIENESTAR, G_CARGA, G_CONTEXTO, G_READINESS
from validacion import (RAIZ, cargar_ejemplos, metricas, particion_temporal,
                        particiones_por_jugadora)

# En macOS, numpy lanza avisos falsos ("overflow/divide by zero encountered in
# matmul") al multiplicar matrices. Se ha comprobado que los resultados son
# finitos e idénticos a calcularlos por otra vía, así que se silencian.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

FIGURAS = RAIZ / "reports" / "figures"
TABLAS = RAIZ / "reports" / "tablas"
AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#898781"
GRIS_CLARO, TINTA = "#e1e0d9", "#52514e"

CONJUNTOS = {
    "Ridge A: readiness": G_READINESS + G_CONTEXTO,
    "Ridge B: + bienestar": G_READINESS + G_CONTEXTO + G_BIENESTAR,
    "Ridge C: + carga": G_READINESS + G_CONTEXTO + G_BIENESTAR + G_CARGA,
}
REFERENCIAS = {"Persistencia": "readiness", "Media de 7 días": "readiness_m7"}
ORDEN = list(REFERENCIAS) + list(CONJUNTOS)


def crear_modelo():
    """Relleno de vacíos con la mediana -> escalado -> Ridge.
    El escalado pone todas las características en la misma unidad, así sus
    coeficientes son comparables entre sí."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))


def evaluar(ejemplos: pd.DataFrame, particiones) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entrena y evalúa cada variante en cada partición, junto a las dos
    referencias, sobre exactamente las mismas filas de prueba."""
    filas, detalle = [], []
    for nombre, idx_entrenamiento, idx_prueba in particiones:
        entrenamiento, prueba = ejemplos.iloc[idx_entrenamiento], ejemplos.iloc[idx_prueba]
        real = prueba["objetivo"].to_numpy()
        predicciones = {ref: prueba[col].to_numpy() for ref, col in REFERENCIAS.items()}
        for variante, columnas in CONJUNTOS.items():
            modelo = crear_modelo().fit(entrenamiento[columnas], entrenamiento["objetivo"])
            # La readiness va de 1 a 10: se recorta por si la recta se sale
            predicciones[variante] = np.clip(modelo.predict(prueba[columnas]), 1, 10)
        for metodo, prediccion in predicciones.items():
            filas.append({"particion": nombre, "metodo": metodo, **metricas(real, prediccion)})
            detalle.append(pd.DataFrame({
                "metodo": metodo, "jugadora": prueba["jugadora"].to_numpy(),
                "hoy": prueba["readiness"].to_numpy(), "real": real,
                "error_abs": np.abs(real - prediccion)}))
    return pd.DataFrame(filas), pd.concat(detalle, ignore_index=True)


def comparar(cv: pd.DataFrame, detalle: pd.DataFrame, a: str, b: str) -> dict:
    """¿La diferencia entre dos métodos es real o podría ser azar?

    - t-test pareado sobre el MAE de las 5 particiones (como en la Práctica 1).
      Con solo 5 valores tiene poca potencia: le cuesta detectar diferencias.
    - Wilcoxon pareado sobre el MAE de cada jugadora (50 valores): no supone
      distribución normal y tiene más datos.
    p < 0,05 = diferencia estadísticamente significativa.
    """
    por_particion = cv.pivot(index="particion", columns="metodo", values="MAE")
    por_jugadora = detalle.pivot_table(index="jugadora", columns="metodo", values="error_abs", aggfunc="mean")
    return {
        "comparacion": f"{a}  vs  {b}",
        "dif_MAE": por_particion[a].mean() - por_particion[b].mean(),
        "p_ttest_particiones": stats.ttest_rel(por_particion[a], por_particion[b]).pvalue,
        "p_wilcoxon_jugadoras": stats.wilcoxon(por_jugadora[a], por_jugadora[b]).pvalue,
        "jugadoras_en_que_gana_a": int((por_jugadora[a] < por_jugadora[b]).sum()),
    }


def coeficientes(ejemplos: pd.DataFrame, columnas: list[str]) -> pd.Series:
    """Coeficientes del modelo entrenado con todos los ejemplos. Al estar las
    entradas escaladas, un coeficiente dice cuántos puntos cambia la predicción
    cuando esa característica sube una desviación típica."""
    modelo = crear_modelo().fit(ejemplos[columnas], ejemplos["objetivo"])
    return pd.Series(modelo[-1].coef_, index=columnas)


def estilo() -> None:
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": TINTA,
        "xtick.color": GRIS, "ytick.color": TINTA, "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True, "legend.frameon": False})


def fig_comparacion(resumen: pd.DataFrame) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    # Gris = referencias sin aprendizaje; azul = modelos
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
                   labels=["Referencia", "Modelo"], loc="lower right", fontsize=8)
    fig.suptitle("Ridge frente a las referencias, validación por jugadoras (media de 5 particiones ± desv. típica)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURAS / "11_ridge_vs_baselines.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_coeficientes(coef: pd.Series, n: int = 15) -> None:
    top = coef.reindex(coef.abs().sort_values(ascending=False).index)[:n][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    # Azul = empuja la predicción hacia arriba; naranja = hacia abajo
    ax.barh(top.index, top.values, color=[AZUL if v > 0 else NARANJA for v in top.values], height=0.65)
    ax.axvline(0, color="#c3c2b7", linewidth=1)
    ax.set_xlabel("Coeficiente (puntos de readiness por desviación típica de la entrada)")
    ax.set_title(f"Ridge C: las {n} características con más peso")
    ax.grid(axis="x", color=GRIS_CLARO, linewidth=0.8)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=AZUL), plt.Rectangle((0, 0), 1, 1, color=NARANJA)],
              labels=["Sube la predicción", "Baja la predicción"], loc="lower right")
    fig.savefig(FIGURAS / "12_ridge_coeficientes.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def resumir(cv: pd.DataFrame) -> pd.DataFrame:
    r = cv.groupby("metodo")[["MAE", "RMSE", "R2"]].agg(["mean", "std"])
    r.columns = [f"{m}_{'media' if e == 'mean' else 'desv'}" for m, e in r.columns]
    return r.loc[ORDEN]


def main() -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    TABLAS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220, "display.float_format", "{:.4f}".format)
    estilo()
    ejemplos = cargar_ejemplos()

    # --- Validación por jugadoras ------------------------------------------------
    cv, detalle = evaluar(ejemplos, particiones_por_jugadora(ejemplos))
    resumen = resumir(cv)
    print("=== VALIDACIÓN POR JUGADORAS (5 particiones) ===")
    print(resumen.to_string())
    print("\nMAE por partición:")
    print(cv.pivot(index="particion", columns="metodo", values="MAE")[ORDEN].to_string())

    d = detalle.assign(tipo=np.where(detalle["real"] == detalle["hoy"], "Mañana = hoy", "Mañana ≠ hoy"),
                       nivel=pd.cut(detalle["real"], [0, 4, 7, 10], labels=["Baja (1-4)", "Media (5-7)", "Alta (8-10)"]))
    desglose = pd.concat([
        d.pivot_table(index="metodo", columns="tipo", values="error_abs", aggfunc="mean"),
        d.pivot_table(index="metodo", columns="nivel", values="error_abs", aggfunc="mean", observed=True),
    ], axis=1).loc[ORDEN]
    print("\nMAE según el tipo de día:")
    print(desglose.to_string())

    pruebas = pd.DataFrame([
        comparar(cv, detalle, "Ridge C: + carga", "Media de 7 días"),
        comparar(cv, detalle, "Ridge C: + carga", "Persistencia"),
        comparar(cv, detalle, "Ridge B: + bienestar", "Ridge A: readiness"),
        comparar(cv, detalle, "Ridge C: + carga", "Ridge B: + bienestar"),
    ]).set_index("comparacion")
    print("\nComparaciones estadísticas (dif_MAE negativa = gana el primero):")
    print(pruebas.to_string())

    # --- Comprobación temporal ---------------------------------------------------
    temporal, _ = evaluar(ejemplos, particion_temporal(ejemplos))
    temporal = temporal.set_index("metodo").loc[ORDEN, ["MAE", "RMSE", "R2"]]
    print("\n=== COMPROBACIÓN TEMPORAL (entrenar 2020, evaluar 2021) ===")
    print(temporal.to_string())

    # --- Coeficientes ------------------------------------------------------------
    coef = coeficientes(ejemplos, CONJUNTOS["Ridge C: + carga"])
    print("\nCoeficientes con más peso (Ridge C):")
    print(coef.reindex(coef.abs().sort_values(ascending=False).index)[:15].to_string())

    resumen.to_csv(TABLAS / "ridge_resumen.csv", float_format="%.4f")
    cv.to_csv(TABLAS / "ridge_por_particion.csv", index=False, float_format="%.4f")
    desglose.to_csv(TABLAS / "ridge_por_tipo_de_dia.csv", float_format="%.4f")
    pruebas.to_csv(TABLAS / "ridge_pruebas_estadisticas.csv", float_format="%.5f")
    temporal.to_csv(TABLAS / "ridge_temporal.csv", float_format="%.4f")
    coef.rename("coeficiente").to_csv(TABLAS / "ridge_coeficientes.csv", float_format="%.4f")
    fig_comparacion(resumen)
    fig_coeficientes(coef)


if __name__ == "__main__":
    main()
