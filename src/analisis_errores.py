"""
Análisis de errores de los modelos de Altair.

No entrena nada: usa reports/tablas/predicciones_por_jugadora.csv, que genera
contraste_jugadoras.py repitiendo la validación por jugadoras de Altair (una
partición por jugadora). Cada predicción es, por tanto, de una jugadora que el
modelo no había visto.

Preguntas:
  1. ¿En qué jugadoras acierta más y menos el modelo, y por qué?
  2. ¿Empeora cuando faltan datos de los días anteriores?
  3. ¿Cuánto falla según lo grande que sea el cambio de un día a otro?
  4. ¿Cómo falla según la franja de readiness (baja, media, alta)?
  5. ¿Cuántas predicciones quedan a 1 punto o menos del valor real?

Se analiza el Random Forest (el modelo explicado con SHAP) frente a la mejor
referencia (la media de 7 días).

Uso (desde la carpeta del proyecto), después de contraste_jugadoras.py:
    .venv/bin/python src/analisis_errores.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import RAIZ, cargar_ejemplos

FIGURAS = RAIZ / "reports" / "figures"
TABLAS = RAIZ / "reports" / "tablas"
AZUL, NARANJA, GRIS, GRIS_CLARO, TINTA = "#2a78d6", "#eb6834", "#898781", "#e1e0d9", "#52514e"
METODOS = ["Media global", "Media propia", "Persistencia", "Media de 7 días", "Ridge", "Random Forest"]
MODELO, BASE = "Random Forest", "Media de 7 días"


def cargar() -> pd.DataFrame:
    """Une las predicciones con el contexto de cada ejemplo y calcula el error
    absoluto de cada método."""
    pred = pd.read_csv(TABLAS / "predicciones_por_jugadora.csv", parse_dates=["fecha"])
    contexto = cargar_ejemplos()[["jugadora", "fecha", "equipo", "temporada", "readiness_hoy",
                                  "tiene_bienestar_ayer", "dias_datos_7d"]]
    d = pred.merge(contexto, on=["jugadora", "fecha"], validate="one_to_one")
    for m in METODOS:
        d[f"err_{m}"] = (d[m] - d["real"]).abs()
    return d


def mae_por(d: pd.DataFrame, grupo, metodos=METODOS) -> pd.DataFrame:
    """MAE de cada método dentro de cada grupo, con el nº de ejemplos."""
    t = d.groupby(grupo, observed=True)[[f"err_{m}" for m in metodos]].mean()
    t.columns = metodos
    t.insert(0, "n", d.groupby(grupo, observed=True).size())
    return t


def por_jugadora(d: pd.DataFrame) -> pd.DataFrame:
    t = mae_por(d, "jugadora")
    t["equipo"] = d.groupby("jugadora")["equipo"].first()
    # Variabilidad: cuánto oscila la readiness de esa jugadora
    t["desv_readiness"] = d.groupby("jugadora")["real"].std()
    t["pct_sin_cambio"] = 100 * (d["real"] == d["readiness_hoy"]).groupby(d["jugadora"]).mean()
    t["mejora"] = t[BASE] - t[MODELO]   # positivo = el modelo gana a la referencia
    return t


def figura(jug: pd.DataFrame, cambio: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": TINTA,
        "xtick.color": GRIS, "ytick.color": GRIS, "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True, "legend.frameon": False})
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.2))

    # 1) Modelo frente a referencia, una jugadora por punto
    tope = max(jug[MODELO].max(), jug[BASE].max()) * 1.05
    a1.plot([0, tope], [0, tope], color="#c3c2b7", linewidth=1, linestyle="--")
    for equipo, color in (("A", AZUL), ("B", NARANJA)):
        s = jug[jug["equipo"] == equipo]
        a1.scatter(s[BASE], s[MODELO], s=np.sqrt(s["n"]) * 3, color=color, alpha=0.75,
                   edgecolors="white", linewidths=1, label=f"Equipo {equipo}")
    a1.text(tope * 0.97, tope * 0.05, "Por debajo de la línea:\ngana el modelo", ha="right",
            fontsize=8, color=TINTA)
    a1.set_xlabel(f"MAE de la {BASE.lower()}")
    a1.set_ylabel(f"MAE del {MODELO}")
    a1.set_title("Error por jugadora (tamaño = nº de ejemplos)")
    a1.grid(color=GRIS_CLARO, linewidth=0.8)
    a1.legend(loc="upper left", fontsize=8)

    # 2) El error depende de lo variable que sea la jugadora
    a2.scatter(jug["desv_readiness"], jug[MODELO], s=np.sqrt(jug["n"]) * 3, color=AZUL, alpha=0.75,
               edgecolors="white", linewidths=1)
    r = jug["desv_readiness"].corr(jug[MODELO])
    a2.set_xlabel("Desviación típica de la readiness de la jugadora")
    a2.set_ylabel(f"MAE del {MODELO}")
    a2.set_title(f"A más variable la jugadora, más error (r = {r:.2f})".replace(".", ","))
    a2.grid(color=GRIS_CLARO, linewidth=0.8)

    # 3) Error según el tamaño del cambio
    x = np.arange(len(cambio))
    a3.bar(x - 0.2, cambio[BASE], width=0.38, color=GRIS, label=BASE)
    a3.bar(x + 0.2, cambio[MODELO], width=0.38, color=AZUL, label=MODELO)
    a3.set_xticks(x)
    a3.set_xticklabels([f"{k}\n({100 * n / cambio['n'].sum():.0f} %)" for k, n in cambio["n"].items()])
    a3.set_xlabel("Cambio de hoy a mañana, en puntos (% de ejemplos)")
    a3.set_ylabel("MAE")
    a3.set_title("Error según el tamaño del cambio")
    a3.grid(axis="y", color=GRIS_CLARO, linewidth=0.8)
    a3.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURAS / "14_analisis_errores.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
    d = cargar()
    print(f"Ejemplos analizados: {len(d)}\n")

    # 1. Por jugadora
    jug = por_jugadora(d)
    print("=== 1. POR JUGADORA ===")
    print(f"MAE del {MODELO}: de {jug[MODELO].min():.2f} a {jug[MODELO].max():.2f} "
          f"(mediana {jug[MODELO].median():.2f})")
    print(f"Gana a la {BASE.lower()} en {(jug['mejora'] > 0).sum()} de {len(jug)} jugadoras")
    print("Correlación del MAE del modelo con:")
    print(f"  variabilidad de la jugadora (desv. típica): {jug[MODELO].corr(jug['desv_readiness']):+.2f}")
    print(f"  % de días sin cambio:                       {jug[MODELO].corr(jug['pct_sin_cambio']):+.2f}")
    print(f"  nº de ejemplos de la jugadora:              {jug[MODELO].corr(jug['n']):+.2f}")
    cols = ["n", BASE, MODELO, "desv_readiness", "pct_sin_cambio"]
    print("Las 5 con más error:")
    print(jug.sort_values(MODELO, ascending=False)[cols].head().to_string())
    print("Las 5 con menos error:")
    print(jug.sort_values(MODELO)[cols].head().to_string())

    # 2. Por huecos
    print("\n=== 2. SEGÚN LOS DATOS DISPONIBLES ===")
    ayer = mae_por(d, d["tiene_bienestar_ayer"].map({1: "Con cuestionario ayer", 0: "Sin cuestionario ayer"}))
    semana = mae_por(d, pd.cut(d["dias_datos_7d"], [0, 3, 6, 7], labels=["1-3 días", "4-6 días", "7 días"]))
    print(ayer.to_string())
    print(semana.rename_axis("Días con datos en la última semana").to_string())

    # 3. Por tamaño del cambio
    print("\n=== 3. SEGÚN EL TAMAÑO DEL CAMBIO ===")
    tam = ((d["real"] - d["readiness_hoy"]).abs().clip(upper=3).astype(int)
           .map({0: "0", 1: "1", 2: "2", 3: "3 o más"}))
    cambio = mae_por(d, tam)
    print(cambio.to_string())

    # 4. Por franja de readiness de mañana (las mismas que el proceso 06 de Altair)
    print("\n=== 4. SEGÚN LA FRANJA DE READINESS DE MAÑANA ===")
    franja = pd.cut(d["real"], [0, 4, 7, 10], labels=["Baja (1-4)", "Media (5-7)", "Alta (8-10)"])
    franjas = mae_por(d, franja)
    print(franjas.to_string())
    # Sesgo: en los días bajos, ¿el modelo se queda corto (predice demasiado alto)?
    sesgo = (d[MODELO] - d["real"]).groupby(franja, observed=True).mean()
    print(f"Error medio con signo del {MODELO} (positivo = predice de más):")
    print(sesgo.to_string())

    # 5. Precisión con margen
    print("\n=== 5. PREDICCIONES CERCA DEL VALOR REAL ===")
    margen = pd.DataFrame({
        "A 0,5 puntos o menos (%)": [100 * (d[f"err_{m}"] <= 0.5).mean() for m in METODOS],
        "A 1 punto o menos (%)": [100 * (d[f"err_{m}"] <= 1).mean() for m in METODOS],
        "Fallo de más de 2 puntos (%)": [100 * (d[f"err_{m}"] > 2).mean() for m in METODOS],
    }, index=METODOS)
    print(margen.round(1).to_string())

    # 6. Otros cortes
    print("\n=== 6. POR EQUIPO Y TEMPORADA ===")
    print(mae_por(d, "equipo", [BASE, MODELO]).to_string())
    print(mae_por(d, "temporada", [BASE, MODELO]).to_string())

    jug.to_csv(TABLAS / "errores_por_jugadora.csv", float_format="%.4f")
    pd.concat([ayer, semana]).to_csv(TABLAS / "errores_por_datos_disponibles.csv", float_format="%.4f")
    cambio.to_csv(TABLAS / "errores_por_cambio.csv", float_format="%.4f")
    franjas.assign(sesgo_rf=sesgo).to_csv(TABLAS / "errores_por_franja.csv", float_format="%.4f")
    margen.to_csv(TABLAS / "errores_margen.csv", float_format="%.2f")
    figura(jug, cambio)


if __name__ == "__main__":
    main()
