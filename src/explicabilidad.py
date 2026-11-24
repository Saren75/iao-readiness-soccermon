"""
Explicabilidad con SHAP del modelo de Altair.

QUÉ ES SHAP
Para cada predicción, SHAP reparte entre las características la diferencia
entre esa predicción y la predicción media. Cada característica recibe un valor
en puntos de readiness: positivo si ha subido la predicción, negativo si la ha
bajado. Siempre se cumple:

    predicción = valor base (predicción media) + suma de los valores SHAP

QUÉ MODELO SE EXPLICA
El mismo que explica Altair en su proceso 06: el Random Forest entrenado con la
temporada 2020 (proceso 05), aplicado a la temporada 2021. Aquí se entrena con
la misma configuración (modelo_altair.py). Altair solo da una importancia global
de cada variable; SHAP dice además en qué sentido empuja la predicción y explica
cada caso concreto.

Para que tarde poco se explican 3.000 días de 2021 elegidos al azar (semilla fija).
Para Random Forest, SHAP se calcula de forma exacta (TreeExplainer).

Uso (desde la carpeta del proyecto); tarda menos de un minuto:
    .venv/bin/python src/explicabilidad.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from features import G_BIENESTAR, G_CARGA, G_CONTEXTO, G_READINESS, OBJETIVO, RAIZ, TODAS, cargar_ejemplos
from modelo_altair import SEMILLA, random_forest_altair

FIGURAS = RAIZ / "reports" / "figures"
TABLAS = RAIZ / "reports" / "tablas"
N_EXPLICADOS = 3000

BLOQUES = {"Historial de readiness": G_READINESS, "Contexto": G_CONTEXTO,
           "Bienestar": G_BIENESTAR, "Carga": G_CARGA}
COLOR_BLOQUE = {"Historial de readiness": "#2a78d6", "Contexto": "#eda100",
                "Bienestar": "#1baf7a", "Carga": "#eb6834"}
BLOQUE_DE = {c: b for b, cols in BLOQUES.items() for c in cols}
AZUL, NARANJA, GRIS, GRIS_CLARO, TINTA = "#2a78d6", "#eb6834", "#898781", "#e1e0d9", "#52514e"


def entrenar_y_explicar(ejemplos: pd.DataFrame):
    """Entrena con 2020, predice 2021 y calcula SHAP sobre una muestra de 2021."""
    entrenamiento = ejemplos[ejemplos["temporada"] == 2020]
    prueba = ejemplos[ejemplos["temporada"] == 2021].reset_index(drop=True)
    modelo = random_forest_altair().fit(entrenamiento[TODAS], entrenamiento[OBJETIVO])

    # Error en 2021, para comprobar que equivale al Random Forest de Altair
    error = prueba[OBJETIVO].to_numpy() - modelo.predict(prueba[TODAS])
    metricas = {"MAE": np.abs(error).mean(), "RMSE": np.sqrt((error ** 2).mean())}

    rellenar, bosque = modelo[0], modelo[-1]
    muestra = prueba.sample(N_EXPLICADOS, random_state=SEMILLA).reset_index(drop=True)
    x = pd.DataFrame(rellenar.transform(muestra[TODAS]), columns=TODAS)
    explicador = shap.TreeExplainer(bosque)
    valores = pd.DataFrame(explicador.shap_values(x.to_numpy()), columns=TODAS)
    base = float(np.ravel(explicador.expected_value)[0])
    prediccion = bosque.predict(x.to_numpy())
    # Propiedad fundamental de SHAP: base + suma de valores = predicción
    assert np.allclose(base + valores.sum(axis=1), prediccion, atol=1e-6)

    info = muestra[["jugadora", "fecha", OBJETIVO]].rename(columns={OBJETIVO: "real"})
    info["prediccion"], info["valor_base"] = prediccion, base
    return metricas, valores, x, info


def estilo() -> None:
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": TINTA,
        "xtick.color": GRIS, "ytick.color": TINTA, "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True, "legend.frameon": False})


def fig_importancia(importancia: pd.Series, por_bloque: pd.Series, n: int = 15) -> None:
    """Importancia global = media del valor absoluto de SHAP: cuántos puntos
    mueve la predicción esa característica, de media, en uno u otro sentido."""
    top = importancia[:n][::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    a1.barh(top.index, top.values, color=[COLOR_BLOQUE[BLOQUE_DE[c]] for c in top.index], height=0.65)
    a1.set_xlabel("Importancia (media de |SHAP|, en puntos de readiness)")
    a1.set_title(f"Las {n} características más influyentes")
    a1.grid(axis="x", color=GRIS_CLARO, linewidth=0.8)
    a1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOR_BLOQUE.values()],
              labels=list(COLOR_BLOQUE), loc="lower right", fontsize=8)
    b = por_bloque.sort_values()
    barras = a2.barh(b.index, 100 * b / b.sum(), color=[COLOR_BLOQUE[k] for k in b.index], height=0.6)
    a2.bar_label(barras, fmt="%.0f %%", padding=3, fontsize=9, color=TINTA)
    a2.set_xlim(0, 100)
    a2.set_xlabel("% de la importancia total")
    a2.set_title("Importancia por bloque")
    a2.grid(axis="x", color=GRIS_CLARO, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(FIGURAS / "15_shap_importancia.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_dependencia(valores: pd.DataFrame, x: pd.DataFrame, caracteristicas: list[str]) -> None:
    """Valor de la característica (eje X) frente a su efecto en la predicción
    (eje Y): muestra la forma de la relación que ha aprendido el modelo."""
    fig, ejes = plt.subplots(1, len(caracteristicas), figsize=(4 * len(caracteristicas), 3.8))
    for ax, c in zip(ejes, caracteristicas):
        ax.scatter(x[c], valores[c], s=6, alpha=0.25, color=COLOR_BLOQUE[BLOQUE_DE[c]], linewidths=0)
        ax.axhline(0, color="#c3c2b7", linewidth=1)
        ax.set_xlabel(c)
        ax.set_title(c, fontsize=10)
        ax.grid(color=GRIS_CLARO, linewidth=0.8)
    ejes[0].set_ylabel("Efecto en la predicción (valor SHAP)")
    fig.suptitle("Cómo cambia la predicción según el valor de cada característica",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURAS / "16_shap_dependencia.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_caso(valores: pd.DataFrame, x: pd.DataFrame, info: pd.DataFrame, i: int, nombre: str, n: int = 8) -> None:
    """Explicación de UNA predicción: qué características la han subido o bajado."""
    fila = valores.iloc[i]
    top = fila.reindex(fila.abs().sort_values(ascending=False).index)[:n]
    etiquetas = [f"{c} = {x.iloc[i][c]:.3g}" for c in top.index] + [f"Las otras {len(fila) - n}"]
    aportes = list(top.values) + [fila.sum() - top.sum()]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(aportes))[::-1]
    barras = ax.barh(y, aportes, color=[AZUL if v > 0 else NARANJA for v in aportes], height=0.65)
    ax.bar_label(barras, labels=[f"{v:+.2f}" for v in aportes], padding=3, fontsize=8, color=TINTA)
    ax.set_yticks(y)
    ax.set_yticklabels(etiquetas)
    ax.axvline(0, color="#c3c2b7", linewidth=1)
    ax.margins(x=0.15)
    ax.set_xlabel("Aportación a la predicción (puntos de readiness)")
    d = info.iloc[i]
    ax.set_title(f"Jugadora {d['jugadora']}, {pd.Timestamp(d['fecha']):%d-%m-%Y}: "
                 f"predicción {d['prediccion']:.2f}, real {d['real']:.0f}", fontsize=11)
    ax.text(0, -0.2, f"Valor base (predicción media): {d['valor_base']:.2f}. "
                     "Azul sube la predicción, naranja la baja.",
            transform=ax.transAxes, fontsize=8, color=TINTA)
    ax.grid(axis="x", color=GRIS_CLARO, linewidth=0.8)
    fig.savefig(FIGURAS / nombre, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    TABLAS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.float_format", "{:.4f}".format)
    estilo()

    metricas, valores, x, info = entrenar_y_explicar(cargar_ejemplos())
    print("Random Forest con la configuración de Altair, entrenado con 2020 y evaluado en 2021:")
    print(f"  MAE {metricas['MAE']:.3f}  ·  RMSE {metricas['RMSE']:.3f}   (Altair: MAE 0,667 · RMSE 0,960)")

    importancia = valores.abs().mean().sort_values(ascending=False)
    por_bloque = importancia.groupby(BLOQUE_DE).sum().reindex(list(BLOQUES))
    print("\nImportancia global (media de |SHAP|), 15 primeras:")
    print(importancia[:15].to_string())
    print("\nImportancia por bloque (% del total):")
    print((100 * por_bloque / por_bloque.sum()).round(1).to_string())
    # Sentido: correlación entre el valor de la característica y su SHAP
    sentido = pd.Series({c: np.corrcoef(x[c], valores[c])[0, 1] for c in importancia.index[:15]})
    print("\nSentido (+ = a más valor, más readiness predicha):")
    print(sentido.round(2).to_string())

    fig_importancia(importancia, por_bloque)
    fig_dependencia(valores, x, ["readiness_media_7d", "readiness_hoy", "carga_hoy"])
    # Dos casos con readiness real baja: el mejor acierto y el peor fallo
    malos = info[info["real"] <= 4]
    acierto = (malos["prediccion"] - malos["real"]).abs().idxmin()
    fallo = (malos["prediccion"] - malos["real"]).idxmax()
    fig_caso(valores, x, info, acierto, "17_shap_caso_dia_malo_acertado.png")
    fig_caso(valores, x, info, fallo, "18_shap_caso_dia_malo_fallado.png")
    print("\nCasos individuales:")
    print(info.loc[[acierto, fallo], ["jugadora", "fecha", "real", "prediccion"]].to_string())

    pd.DataFrame({"importancia": importancia, "bloque": pd.Series(BLOQUE_DE)}).sort_values(
        "importancia", ascending=False).to_csv(TABLAS / "shap_importancia.csv", float_format="%.5f")
    (100 * por_bloque / por_bloque.sum()).rename("pct_importancia").to_csv(
        TABLAS / "shap_importancia_por_bloque.csv", float_format="%.2f")
    pd.Series(metricas).rename("valor").to_csv(TABLAS / "shap_modelo_temporal.csv", float_format="%.4f")


if __name__ == "__main__":
    main()
