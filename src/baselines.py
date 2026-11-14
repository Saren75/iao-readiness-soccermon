"""
Fase 4 - Baselines: las referencias que todo modelo tendrá que superar.

Un baseline es una forma de predecir tan simple que no necesita aprendizaje
automático. Si un modelo no la mejora, no aporta nada.

Baselines (todos predicen la readiness de mañana):
  - Media global:      la media de la readiness en los datos de entrenamiento.
                       Es el único que "aprende" algo del entrenamiento.
  - Media propia:      la media de todos los cuestionarios de esa jugadora
                       hasta hoy (su nivel habitual).
  - Media de 7 días:   la media de sus últimos 7 días.
  - Persistencia:      mañana estará igual que hoy.

Los tres últimos solo usan el pasado de la propia jugadora, así que valen
también para una jugadora que el sistema no ha visto nunca.

Uso (desde la carpeta "Practica final"):
    .venv/bin/python src/baselines.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validacion import (RAIZ, cargar_ejemplos, metricas, particion_temporal,
                        particiones_por_jugadora)

FIGURAS = RAIZ / "reports" / "figures"
TABLAS = RAIZ / "reports" / "tablas"
AZUL, GRIS_CLARO, GRIS_EJE, TINTA = "#2a78d6", "#e1e0d9", "#898781", "#52514e"

ORDEN = ["Media global", "Media propia", "Media de 7 días", "Persistencia"]


def predecir(entrenamiento: pd.DataFrame, prueba: pd.DataFrame) -> dict:
    """Predicción de cada baseline para las filas de prueba."""
    return {
        "Media global": np.full(len(prueba), entrenamiento["objetivo"].mean()),
        "Media propia": prueba["media_propia"].to_numpy(),
        "Media de 7 días": prueba["media_7d"].to_numpy(),
        "Persistencia": prueba["readiness"].to_numpy(),
    }


def evaluar(ejemplos: pd.DataFrame, particiones) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evalúa cada baseline en cada partición.

    Devuelve dos tablas:
      - por_particion: una fila por (partición, baseline) con sus métricas
      - detalle: una fila por (ejemplo de prueba, baseline) con su error,
        para poder analizar después por jugadora o por tipo de día
    """
    filas, detalle = [], []
    for nombre, idx_entrenamiento, idx_prueba in particiones:
        entrenamiento, prueba = ejemplos.iloc[idx_entrenamiento], ejemplos.iloc[idx_prueba]
        real = prueba["objetivo"].to_numpy()
        for baseline, prediccion in predecir(entrenamiento, prueba).items():
            filas.append({"particion": nombre, "baseline": baseline,
                          "n_prueba": len(prueba),
                          "jugadoras_prueba": prueba["jugadora"].nunique(),
                          **metricas(real, prediccion)})
            detalle.append(pd.DataFrame({
                "baseline": baseline, "jugadora": prueba["jugadora"].to_numpy(),
                "hoy": prueba["readiness"].to_numpy(), "real": real,
                "error_abs": np.abs(real - prediccion)}))
    return pd.DataFrame(filas), pd.concat(detalle, ignore_index=True)


def resumir(por_particion: pd.DataFrame) -> pd.DataFrame:
    """Media y desviación típica de cada métrica entre particiones."""
    r = por_particion.groupby("baseline")[["MAE", "RMSE", "R2"]].agg(["mean", "std"])
    r.columns = [f"{metrica}_{'media' if est == 'mean' else 'desv'}" for metrica, est in r.columns]
    return r.loc[ORDEN]


def por_tipo_de_dia(detalle: pd.DataFrame) -> pd.DataFrame:
    """MAE separando los días en que la readiness cambia de los que no.
    La persistencia acierta siempre cuando no hay cambio, así que su punto
    débil está, por definición, en los días con cambio."""
    d = detalle.assign(tipo=np.where(detalle["real"] == detalle["hoy"],
                                     "Mañana = hoy", "Mañana ≠ hoy"))
    return d.pivot_table(index="baseline", columns="tipo", values="error_abs",
                         aggfunc="mean").loc[ORDEN]


def por_nivel(detalle: pd.DataFrame) -> pd.DataFrame:
    """MAE según la readiness real de mañana: baja (1-4), media (5-7), alta (8-10)."""
    nivel = pd.cut(detalle["real"], [0, 4, 7, 10], labels=["Baja (1-4)", "Media (5-7)", "Alta (8-10)"])
    tabla = detalle.assign(nivel=nivel).pivot_table(
        index="baseline", columns="nivel", values="error_abs", aggfunc="mean", observed=True)
    return tabla.loc[ORDEN]


def figura(resumen: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": TINTA,
        "xtick.color": GRIS_EJE, "ytick.color": TINTA, "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True})
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for ax, metrica in zip(ejes, ("MAE", "RMSE")):
        media, desv = resumen[f"{metrica}_media"], resumen[f"{metrica}_desv"]
        barras = ax.barh(resumen.index, media, xerr=desv, color=AZUL, height=0.6,
                         error_kw={"ecolor": TINTA, "elinewidth": 1.2, "capsize": 3})
        for barra, m, s in zip(barras, media, desv):
            ax.text(m + s + 0.03, barra.get_y() + barra.get_height() / 2,
                    f"{m:.3f}".replace(".", ","), va="center", fontsize=9, color=TINTA)
        ax.set_xlim(0, 1.9)
        ax.set_xlabel(f"{metrica} (puntos de readiness; menos es mejor)")
        ax.set_title(metrica)
        ax.grid(axis="x", color=GRIS_CLARO, linewidth=0.8)
    fig.suptitle("Baselines en validación por jugadoras (media de 5 particiones ± desviación típica)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURAS / "10_baselines.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    TABLAS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)

    ejemplos = cargar_ejemplos()
    print(f"Ejemplos: {len(ejemplos)} de {ejemplos['jugadora'].nunique()} jugadoras\n")

    # --- Validación principal: por jugadoras -----------------------------------
    cv, detalle = evaluar(ejemplos, particiones_por_jugadora(ejemplos))
    resumen = resumir(cv)
    print("=== VALIDACIÓN POR JUGADORAS (5 particiones) ===")
    print(cv.groupby("particion")[["n_prueba", "jugadoras_prueba"]].first().to_string())
    print(resumen.to_string())
    print("\nMAE según si la readiness cambia o no:")
    print(por_tipo_de_dia(detalle).to_string())
    print("\nMAE según la readiness real de mañana:")
    print(por_nivel(detalle).to_string())
    jug = detalle.pivot_table(index="jugadora", columns="baseline", values="error_abs", aggfunc="mean")[ORDEN]
    mejor = jug.idxmin(axis=1).value_counts().reindex(ORDEN, fill_value=0)
    print("\nNº de jugadoras en las que cada baseline es el mejor (por MAE):")
    print(mejor.to_string())

    # --- Comprobación temporal: 2020 -> 2021 -----------------------------------
    temporal, _ = evaluar(ejemplos, particion_temporal(ejemplos))
    temporal = temporal.set_index("baseline").loc[ORDEN, ["n_prueba", "MAE", "RMSE", "R2"]]
    print("\n=== COMPROBACIÓN TEMPORAL (entrenar 2020, evaluar 2021) ===")
    print(temporal.to_string())

    cv.to_csv(TABLAS / "baselines_por_particion.csv", index=False, float_format="%.4f")
    resumen.to_csv(TABLAS / "baselines_resumen.csv", float_format="%.4f")
    temporal.to_csv(TABLAS / "baselines_temporal.csv", float_format="%.4f")
    jug.to_csv(TABLAS / "baselines_por_jugadora.csv", float_format="%.4f")
    figura(resumen)


if __name__ == "__main__":
    main()
