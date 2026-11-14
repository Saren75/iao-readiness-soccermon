"""
Fase 7 - Explicabilidad con SHAP.

QUÉ ES SHAP
Para cada predicción, SHAP reparte entre las características la diferencia
entre esa predicción y la predicción media. Cada característica recibe un
valor (en puntos de readiness): positivo si ha empujado la predicción hacia
arriba, negativo si la ha empujado hacia abajo. Siempre se cumple:

    predicción = valor base (predicción media) + suma de los valores SHAP

POR QUÉ SOBRE RANDOM FOREST
Los tres modelos rinden igual (no hay diferencias significativas). Se explica
Random Forest porque es el mejor en la comprobación temporal y porque para
modelos de árboles SHAP se calcula de forma exacta (TreeExplainer), sin las
aproximaciones que necesitaría una red de neuronas.

CÓMO SE CALCULA SIN HACER TRAMPA
Igual que al evaluar: en cada una de las 5 particiones se entrena con 40
jugadoras y se explican predicciones de las 10 restantes. Así se explica cómo
decide el modelo ante jugadoras que no ha visto. Para que tarde poco se
explican 600 ejemplos al azar por partición (3.000 en total).

Uso (desde la carpeta "Practica final"); tarda uno o dos minutos:
    .venv/bin/python src/explicabilidad.py
"""

import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from features import G_BIENESTAR, G_CARGA, G_CONTEXTO, G_READINESS, TODAS
from validacion import RAIZ, cargar_ejemplos, particiones_por_jugadora

FIGURAS = RAIZ / "reports" / "figures"
TABLAS = RAIZ / "reports" / "tablas"
SEMILLA = 0
EJEMPLOS_POR_PARTICION = 600

# Un color fijo por bloque de características
BLOQUES = {"Historial de readiness": G_READINESS, "Contexto": G_CONTEXTO,
           "Bienestar": G_BIENESTAR, "Carga": G_CARGA}
COLOR_BLOQUE = {"Historial de readiness": "#2a78d6", "Contexto": "#eda100",
                "Bienestar": "#1baf7a", "Carga": "#eb6834"}
BLOQUE_DE = {c: b for b, cols in BLOQUES.items() for c in cols}
AZUL, NARANJA, GRIS, GRIS_CLARO, TINTA = "#2a78d6", "#eb6834", "#898781", "#e1e0d9", "#52514e"


def calcular_shap(ejemplos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Devuelve tres tablas alineadas fila a fila:
       valores SHAP, valores de las características y datos del ejemplo."""
    rng = np.random.default_rng(SEMILLA)
    shap_l, x_l, info_l = [], [], []
    for nombre, idx_entrenamiento, idx_prueba in particiones_por_jugadora(ejemplos):
        entrenamiento = ejemplos.iloc[idx_entrenamiento]
        # Relleno de vacíos ajustado solo con entrenamiento, como en modelos.py
        rellenar = SimpleImputer(strategy="median").fit(entrenamiento[TODAS])
        # Hiperparámetros: la combinación más elegida en la validación anidada
        modelo = RandomForestRegressor(n_estimators=300, min_samples_leaf=20, max_features=1.0,
                                       random_state=SEMILLA, n_jobs=-1)
        modelo.fit(rellenar.transform(entrenamiento[TODAS]), entrenamiento["objetivo"])

        elegidos = np.sort(rng.choice(idx_prueba, size=EJEMPLOS_POR_PARTICION, replace=False))
        prueba = ejemplos.iloc[elegidos]
        x = rellenar.transform(prueba[TODAS])
        explicador = shap.TreeExplainer(modelo)
        valores = explicador.shap_values(x)
        base = float(np.ravel(explicador.expected_value)[0])
        prediccion = modelo.predict(x)
        # Comprobación de la propiedad fundamental de SHAP
        assert np.allclose(base + valores.sum(axis=1), prediccion, atol=1e-6), "SHAP no suma la predicción"

        shap_l.append(pd.DataFrame(valores, columns=TODAS))
        x_l.append(pd.DataFrame(x, columns=TODAS))
        info_l.append(pd.DataFrame({
            "particion": nombre, "jugadora": prueba["jugadora"].to_numpy(),
            "fecha": prueba["fecha"].to_numpy(), "real": prueba["objetivo"].to_numpy(),
            "prediccion": prediccion, "valor_base": base}))
        print(f"  {nombre}: {len(prueba)} ejemplos explicados", flush=True)
    return (pd.concat(shap_l, ignore_index=True), pd.concat(x_l, ignore_index=True),
            pd.concat(info_l, ignore_index=True))


def estilo() -> None:
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": TINTA,
        "xtick.color": GRIS, "ytick.color": TINTA, "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True, "legend.frameon": False})


def fig_importancia(importancia: pd.Series, por_bloque: pd.Series, n: int = 15) -> None:
    """Importancia global = media del valor absoluto de SHAP: cuántos puntos
    mueve la predicción esa característica, de media, en uno u otro sentido."""
    top = importancia.sort_values(ascending=False)[:n][::-1]
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
    """Para cada característica: su valor (eje X) frente a su efecto en la
    predicción (eje Y). Muestra la FORMA de la relación que ha aprendido el modelo."""
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
    """Explicación de UNA predicción: qué características la han subido o
    bajado respecto a la predicción media."""
    fila = valores.iloc[i]
    top = fila.reindex(fila.abs().sort_values(ascending=False).index)[:n]
    resto = fila.sum() - top.sum()
    etiquetas = [f"{c} = {x.iloc[i][c]:.3g}" for c in top.index] + [f"Las otras {len(fila) - n}"]
    aportes = list(top.values) + [resto]
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
                     f"Azul sube la predicción, naranja la baja.",
            transform=ax.transAxes, fontsize=8, color=TINTA)
    ax.grid(axis="x", color=GRIS_CLARO, linewidth=0.8)
    fig.savefig(FIGURAS / nombre, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    TABLAS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.float_format", "{:.4f}".format)
    estilo()

    valores, x, info = calcular_shap(cargar_ejemplos())

    importancia = valores.abs().mean().sort_values(ascending=False)
    por_bloque = importancia.groupby(BLOQUE_DE).sum().reindex(list(BLOQUES))
    print("\nImportancia global (media de |SHAP|), 15 primeras:")
    print(importancia[:15].to_string())
    print("\nImportancia por bloque (% del total):")
    print((100 * por_bloque / por_bloque.sum()).round(1).to_string())
    print(f"\nValor base medio: {info['valor_base'].mean():.3f}")

    # Sentido del efecto: correlación entre el valor de la característica y su SHAP
    sentido = pd.Series({c: np.corrcoef(x[c], valores[c])[0, 1] for c in importancia.index[:15]})
    print("\nSentido (correlación valor-SHAP; + = a más valor, más readiness predicha):")
    print(sentido.round(2).to_string())

    fig_importancia(importancia, por_bloque)
    fig_dependencia(valores, x, ["readiness_m7", "readiness", "carga_diaria"])

    # Dos casos individuales: un día malo bien anticipado y uno no anticipado
    malos = info[info["real"] <= 4]
    acierto = (malos["prediccion"] - malos["real"]).abs().idxmin()
    fallo = (malos["prediccion"] - malos["real"]).idxmax()
    fig_caso(valores, x, info, acierto, "17_shap_caso_dia_malo_acertado.png")
    fig_caso(valores, x, info, fallo, "18_shap_caso_dia_malo_fallado.png")
    print("\nCasos individuales:")
    print(info.loc[[acierto, fallo], ["jugadora", "fecha", "real", "prediccion"]].to_string())

    tabla = pd.DataFrame({"importancia": importancia, "bloque": pd.Series(BLOQUE_DE)}).sort_values(
        "importancia", ascending=False)
    tabla.to_csv(TABLAS / "shap_importancia.csv", float_format="%.5f")
    (100 * por_bloque / por_bloque.sum()).rename("pct_importancia").to_csv(
        TABLAS / "shap_importancia_por_bloque.csv", float_format="%.2f")


if __name__ == "__main__":
    main()
