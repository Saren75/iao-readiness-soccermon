"""
Fase 2 - Análisis exploratorio de datos (AED).

Lee data/processed/jugadora_dia.csv (generada por src/carga.py) y produce:
  - figuras en reports/figures/
  - tablas  en reports/tablas/    (los números que hay detrás de cada figura)
  - un resumen por pantalla con las cifras clave

Uso (desde la carpeta del proyecto):
    .venv/bin/python src/aed.py

Preguntas que responde, en este orden:
  1. ¿Cuántos datos hay por jugadora y cómo se reparten los huecos?
  2. ¿Cómo se distribuye la readiness, en general y por jugadora?
  3. ¿Cuánto se parece la readiness de un día a la de los días anteriores?
  4. ¿Qué variables de hoy se relacionan con la readiness de mañana?
  5. ¿Cuántos ejemplos quedan según lo exigentes que seamos con los huecos?

Nota: aquí NO se entrena nada. Solo se mira, para decidir con datos.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # dibuja a fichero, sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
FIGURAS = RAIZ / "reports" / "figures"
TABLAS = RAIZ / "reports" / "tablas"

# --- Estilo común de las figuras ----------------------------------------------
AZUL, NARANJA = "#2a78d6", "#eb6834"      # equipo A / equipo B, siempre igual
GRIS_CLARO, GRIS_EJE, TINTA = "#e1e0d9", "#898781", "#52514e"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": TINTA, "xtick.color": GRIS_EJE, "ytick.color": GRIS_EJE,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRIS_CLARO, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})

BIENESTAR = ["fatiga", "animo", "sueno_horas", "sueno_calidad", "dolor_muscular", "estres"]
CARGA = ["carga_diaria", "duracion_min", "carga_semanal", "atl", "ctl28", "ctl42",
         "acwr", "monotonia", "strain"]
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def mes_es(fecha) -> str:
    """Etiqueta de fecha en español, p. ej. 'abr 2020'."""
    return f"{MESES[fecha.month - 1]} {fecha.year}"


ETIQUETAS = {
    "readiness": "Readiness", "fatiga": "Fatiga (5 = fresca)", "animo": "Ánimo",
    "sueno_horas": "Horas de sueño", "sueno_calidad": "Calidad del sueño",
    "dolor_muscular": "Dolor muscular (5 = sin dolor)", "estres": "Estrés (5 = sin estrés)",
    "carga_diaria": "Carga del día", "duracion_min": "Minutos entrenados",
    "carga_semanal": "Carga semanal", "atl": "ATL (media 7 días)",
    "ctl28": "CTL (media 28 días)", "ctl42": "CTL (media 42 días)",
    "acwr": "ACWR", "monotonia": "Monotonía", "strain": "Strain",
}


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Correlación de Spearman: la de Pearson calculada sobre los rangos.
    Se usa en vez de Pearson porque las escalas son ordinales (1-5, 1-10)."""
    return a.rank().corr(b.rank())


def centrar_por_jugadora(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Resta a cada valor la media de su jugadora. Así se compara a cada
    jugadora consigo misma y se elimina el efecto de que unas puntúen
    siempre más alto que otras."""
    return df.groupby("jugadora")[columnas].transform(lambda x: x - x.mean())


def guardar(fig, nombre: str) -> None:
    fig.savefig(FIGURAS / nombre, facecolor="white")
    plt.close(fig)


# ==============================================================================
# 1. Cantidad de datos y huecos
# ==============================================================================
def resumen_jugadoras(t: pd.DataFrame) -> pd.DataFrame:
    """Una fila por jugadora: cuántos días tiene, en qué periodo y con qué
    constancia contesta dentro de ese periodo."""
    con = t[t["tiene_bienestar"] == 1]
    r = con.groupby("jugadora").agg(
        equipo=("equipo", "first"),
        dias_con_readiness=("fecha", "size"),
        primer_dia=("fecha", "min"),
        ultimo_dia=("fecha", "max"),
        readiness_media=("readiness", "mean"),
        readiness_desv=("readiness", "std"),
    )
    # Periodo activo: del primer al último cuestionario. La cobertura dentro de
    # ese periodo es más justa que sobre los 731 días, porque no todas
    # estuvieron en el equipo las dos temporadas.
    r["dias_periodo_activo"] = (r["ultimo_dia"] - r["primer_dia"]).dt.days + 1
    r["cobertura_periodo_%"] = 100 * r["dias_con_readiness"] / r["dias_periodo_activo"]
    # Pares utilizables: días con readiness hoy Y mañana (lo que se podrá predecir)
    r["pares_t_t1"] = t.dropna(subset=["readiness", "readiness_sig"]).groupby("jugadora").size()
    r["pares_t_t1"] = r["pares_t_t1"].fillna(0).astype(int)
    return r.sort_values("dias_con_readiness", ascending=False)


def fig_dias_por_jugadora(r: pd.DataFrame) -> None:
    r = r.sort_values("dias_con_readiness")
    fig, ax = plt.subplots(figsize=(7, 8.5))
    colores = r["equipo"].map({"A": AZUL, "B": NARANJA})
    ax.barh(r.index, r["dias_con_readiness"], color=colores, height=0.75)
    ax.set_xlabel("Días con cuestionario de bienestar (de 731 posibles)")
    ax.set_title("Días con datos por jugadora")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7)
    ax.margins(y=0.01)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=AZUL),
                       plt.Rectangle((0, 0), 1, 1, color=NARANJA)],
              labels=["Equipo A", "Equipo B"], loc="lower right")
    guardar(fig, "01_dias_por_jugadora.png")


def fig_calendario(t: pd.DataFrame, r: pd.DataFrame) -> None:
    """Mapa jugadora x día: una celda oscura = ese día hay cuestionario."""
    orden = r.sort_values(["equipo", "dias_con_readiness"], ascending=[True, False]).index
    m = t.pivot(index="jugadora", columns="fecha", values="tiene_bienestar").loc[orden]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(m.values, aspect="auto", interpolation="nearest",
              cmap=matplotlib.colors.ListedColormap(["#f0efec", AZUL]))
    meses = pd.date_range(m.columns[0], m.columns[-1], freq="3MS")
    ax.set_xticks([m.columns.get_loc(d) for d in meses])
    ax.set_xticklabels([mes_es(d) for d in meses], rotation=45, ha="right")
    ax.set_yticks(range(len(orden)))
    ax.set_yticklabels(orden, fontsize=6)
    ax.axhline((r["equipo"] == "A").sum() - 0.5, color="white", linewidth=3)
    ax.grid(False)
    ax.set_title("Cuándo contesta cada jugadora (azul = hay cuestionario ese día)")
    guardar(fig, "02_calendario_respuestas.png")


def fig_respuesta_mensual(t: pd.DataFrame) -> pd.DataFrame:
    """Porcentaje de jugadoras que contestan, mes a mes y por equipo."""
    mes = t.assign(mes=t["fecha"].dt.to_period("M").dt.to_timestamp())
    tasa = 100 * mes.pivot_table(index="mes", columns="equipo", values="tiene_bienestar", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for equipo, color in (("A", AZUL), ("B", NARANJA)):
        ax.plot(tasa.index, tasa[equipo], color=color, linewidth=2, label=f"Equipo {equipo}")
        ax.annotate(f"Equipo {equipo}", (tasa.index[-1], tasa[equipo].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", color=TINTA, fontsize=9)
    marcas = tasa.index[::3]
    ax.set_xticks(marcas)
    ax.set_xticklabels([mes_es(d) for d in marcas])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% de días-jugadora con cuestionario")
    ax.set_title("Tasa de respuesta por mes")
    ax.legend(loc="upper left", ncols=2)
    guardar(fig, "03_respuesta_mensual.png")
    return tasa


# ==============================================================================
# 2. Distribución de la readiness
# ==============================================================================
def fig_distribucion(t: pd.DataFrame) -> pd.Series:
    frec = 100 * t["readiness"].value_counts(normalize=True).sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    barras = ax.bar(frec.index.astype(int), frec.values, color=AZUL, width=0.8)
    ax.bar_label(barras, fmt="%.1f %%", fontsize=8, color=TINTA, padding=2)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Readiness declarada (1 = nada preparada, 10 = con muchas ganas)")
    ax.set_ylabel("% de cuestionarios")
    ax.set_title("Distribución de la readiness")
    ax.grid(axis="x", visible=False)
    guardar(fig, "04_distribucion_readiness.png")
    return frec


def fig_readiness_por_jugadora(r: pd.DataFrame, media_global: float) -> None:
    """Media de cada jugadora con su dispersión (media ± desviación típica)."""
    r = r.sort_values("readiness_media")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(r))
    for equipo, color in (("A", AZUL), ("B", NARANJA)):
        sel = (r["equipo"] == equipo).values
        ax.errorbar(x[sel], r["readiness_media"][sel], yerr=r["readiness_desv"][sel],
                    fmt="o", color=color, ecolor=color, elinewidth=1.2, markersize=5,
                    alpha=0.9, label=f"Equipo {equipo}")
    ax.axhline(media_global, color=GRIS_EJE, linewidth=1, linestyle="--")
    ax.annotate(f"Media global: {media_global:.2f}", (0, media_global), xytext=(0, 5),
                textcoords="offset points", color=TINTA, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(r.index, rotation=90, fontsize=7)
    ax.set_ylim(1, 10)
    ax.set_ylabel("Readiness (media ± desviación típica)")
    ax.set_title("Cada jugadora tiene su propio nivel habitual")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", ncols=2)
    guardar(fig, "05_readiness_por_jugadora.png")


# ==============================================================================
# 3. Parecido con los días anteriores
# ==============================================================================
def autocorrelacion(t: pd.DataFrame, max_desfase: int = 14) -> pd.DataFrame:
    """Correlación entre la readiness de un día y la de k días antes.
    'global' mezcla dos cosas: que cada jugadora tiene su nivel y que los días
    seguidos se parecen. 'intra' quita lo primero (centra por jugadora)."""
    d = t[["jugadora", "readiness"]].copy()
    d["centrada"] = centrar_por_jugadora(t, ["readiness"])["readiness"]
    filas = []
    for k in range(1, max_desfase + 1):
        # shift dentro de cada jugadora; el calendario es completo, así que
        # la fila k posiciones atrás es exactamente k días antes.
        ant = d.groupby("jugadora")[["readiness", "centrada"]].shift(k)
        ok = d["readiness"].notna() & ant["readiness"].notna()
        filas.append({"desfase_dias": k, "n_pares": int(ok.sum()),
                      "global": d.loc[ok, "readiness"].corr(ant.loc[ok, "readiness"]),
                      "intra_jugadora": d.loc[ok, "centrada"].corr(ant.loc[ok, "centrada"])})
    return pd.DataFrame(filas).set_index("desfase_dias")


def respuestas_repetidas(t: pd.DataFrame) -> pd.DataFrame:
    """Rachas: días seguidos en el calendario con exactamente la misma readiness.
    Sirve para medir cuántas respuestas son 'siempre lo mismo'."""
    c = t[t["tiene_bienestar"] == 1].copy()
    g = c.groupby("jugadora")
    # Empieza racha nueva si cambia el valor o si hay un salto de calendario
    nueva = g["readiness"].diff().ne(0) | g["fecha"].diff().dt.days.ne(1)
    largo = c.groupby(["jugadora", nueva.cumsum()]).size()
    filas = [{"racha_minima_dias": k, "cuestionarios": int(largo[largo >= k].sum()),
              "% de cuestionarios": 100 * largo[largo >= k].sum() / len(c),
              "jugadoras": largo[largo >= k].index.get_level_values(0).nunique()}
             for k in (7, 14, 30)]
    return pd.DataFrame(filas).set_index("racha_minima_dias")


def persistencia_por_jugadora(pares: pd.DataFrame) -> pd.DataFrame:
    """Para cada jugadora: % de días en que mañana = hoy y error medio de
    suponer que mañana será igual que hoy."""
    d = pares.assign(sin_cambio=pares["readiness"] == pares["readiness_sig"],
                     error=(pares["readiness"] - pares["readiness_sig"]).abs())
    return d.groupby("jugadora").agg(pares=("error", "size"),
                                     sin_cambio_pct=("sin_cambio", lambda x: 100 * x.mean()),
                                     mae_persistencia=("error", "mean"))


def fig_autocorrelacion(ac: pd.DataFrame, cambio: pd.Series) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9))
    a1.plot(ac.index, ac["global"], color=AZUL, linewidth=2, marker="o", markersize=5, label="Global")
    a1.plot(ac.index, ac["intra_jugadora"], color=NARANJA, linewidth=2, marker="o", markersize=5,
            label="Dentro de cada jugadora")
    a1.set_ylim(0, 1)
    a1.set_xticks(ac.index)
    a1.set_xlabel("Días de separación")
    a1.set_ylabel("Correlación")
    a1.set_title("Parecido con los días anteriores")
    a1.legend(loc="upper right")

    barras = a2.bar(cambio.index, cambio.values, color=AZUL, width=0.8)
    a2.bar_label(barras, fmt="%.0f %%", fontsize=8, color=TINTA, padding=2)
    a2.set_xticks(cambio.index)
    a2.set_xticklabels(["≤ −3"] + [f"{v:+d}" if v else "0" for v in cambio.index[1:-1]] + ["≥ +3"])
    a2.set_xlabel("Cambio de un día al siguiente (puntos)")
    a2.set_ylabel("% de pares de días")
    a2.set_title("Cuánto cambia de hoy a mañana")
    a2.grid(axis="x", visible=False)
    guardar(fig, "06_autocorrelacion_y_cambio.png")


# ==============================================================================
# 4. Relación de las variables de hoy con la readiness de mañana
# ==============================================================================
def correlaciones_objetivo(t: pd.DataFrame) -> pd.DataFrame:
    variables = ["readiness"] + BIENESTAR + CARGA
    filas = []
    for v in variables:
        d = t[["jugadora", v, "readiness_sig"]].dropna()
        c = centrar_por_jugadora(d, [v, "readiness_sig"])
        filas.append({"variable": v, "n": len(d),
                      "global": spearman(d[v], d["readiness_sig"]),
                      "intra_jugadora": spearman(c[v], c["readiness_sig"])})
    return pd.DataFrame(filas).set_index("variable")


def fig_correlaciones(co: pd.DataFrame) -> None:
    co = co.sort_values("intra_jugadora")
    y = np.arange(len(co))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(y + 0.19, co["global"], height=0.34, color=AZUL, label="Global")
    ax.barh(y - 0.19, co["intra_jugadora"], height=0.34, color=NARANJA, label="Dentro de cada jugadora")
    ax.axvline(0, color="#c3c2b7", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([ETIQUETAS[v] + " (hoy)" for v in co.index])
    ax.set_xlabel("Correlación de Spearman con la readiness de mañana")
    ax.set_title("Qué variables de hoy se relacionan con la readiness de mañana")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    guardar(fig, "07_correlaciones_objetivo.png")


def fig_matriz_correlacion(t: pd.DataFrame) -> pd.DataFrame:
    """Correlación entre las entradas, para ver cuáles son redundantes."""
    variables = ["readiness"] + BIENESTAR + CARGA
    m = t.loc[t["tiene_bienestar"] == 1, variables].rank().corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    # Escala divergente: azul = positiva, naranja = negativa, gris claro = 0
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", [NARANJA, "#f0efec", AZUL])
    im = ax.imshow(m.values, cmap=cmap, vmin=-1, vmax=1)
    cortas = {"ctl28": "CTL 28 días", "ctl42": "CTL 42 días", "atl": "ATL"}
    nombres = [cortas.get(v, ETIQUETAS[v].split(" (")[0]) for v in variables]
    ax.set_xticks(range(len(variables)))
    ax.set_xticklabels(nombres, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(variables)))
    ax.set_yticklabels(nombres, fontsize=8)
    for i in range(len(variables)):
        for j in range(len(variables)):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.2f}".replace("0.", ".").replace("1.00", "1"), ha="center", va="center",
                    fontsize=6.5, color="white" if abs(v) > 0.6 else "#0b0b0b")
    ax.grid(False)
    ax.set_title("Correlación entre variables (Spearman, mismo día)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Correlación")
    guardar(fig, "08_matriz_correlacion.png")
    return m


def fig_series_ejemplo(t: pd.DataFrame, r: pd.DataFrame) -> list[str]:
    """Tres jugadoras con muchos datos y niveles distintos: la de media más
    baja, la intermedia y la más alta entre las que tienen al menos 400 días."""
    cand = r[r["dias_con_readiness"] >= 400].sort_values("readiness_media")
    elegidas = [cand.index[0], cand.index[len(cand) // 2], cand.index[-1]]
    fig, ejes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True, sharey=True)
    for ax, j in zip(ejes, elegidas):
        d = t[t["jugadora"] == j].set_index("fecha")["readiness"]
        color = AZUL if r.loc[j, "equipo"] == "A" else NARANJA
        ax.plot(d.index, d, color=GRIS_CLARO, linewidth=0.8)             # valor diario (con huecos)
        ax.plot(d.index, d.rolling(7, min_periods=4).mean(), color=color, linewidth=2)  # media de 7 días
        ax.set_ylim(1, 10)
        ax.set_ylabel("Readiness")
        ax.set_title(f"Jugadora {j}: media {r.loc[j, 'readiness_media']:.1f}, "
                     f"{r.loc[j, 'dias_con_readiness']} días con datos", fontsize=10)
    marcas = pd.date_range(t["fecha"].min(), t["fecha"].max(), freq="3MS")
    ejes[-1].set_xticks(marcas)
    ejes[-1].set_xticklabels([mes_es(d) for d in marcas])
    fig.suptitle("Evolución de tres jugadoras (gris: valor diario; color: media de 7 días)",
                 x=0.07, ha="left", fontweight="bold")
    fig.tight_layout()
    guardar(fig, "09_series_ejemplo.png")
    return elegidas


# ==============================================================================
# 5. ¿Cuántos ejemplos quedan según lo exigentes que seamos con los huecos?
# ==============================================================================
def ejemplos_disponibles(t: pd.DataFrame) -> pd.DataFrame:
    """Parte de los pares (readiness hoy y mañana) y cuenta cuántos sobreviven
    al exigir, además, datos de días anteriores."""
    hay = t["tiene_bienestar"]
    g = hay.groupby(t["jugadora"])
    ayer, anteayer = g.shift(1), g.shift(2)
    # nº de días con cuestionario entre t-6 y t (ventana de 7 días que acaba hoy)
    en_7 = g.rolling(7, min_periods=1).sum().reset_index(level=0, drop=True)
    en_3 = g.rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    base = t["readiness"].notna() & t["readiness_sig"].notna()
    criterios = {
        "Readiness hoy y mañana (mínimo imprescindible)": base,
        "+ cuestionario ayer (desfase de 1 día)": base & (ayer == 1),
        "+ ayer y anteayer (desfases de 1 y 2 días)": base & (ayer == 1) & (anteayer == 1),
        "+ los 3 últimos días completos": base & (en_3 == 3),
        "+ al menos 4 de los últimos 7 días": base & (en_7 >= 4),
        "+ los 7 últimos días completos": base & (en_7 == 7),
    }
    filas = [{"criterio": k, "ejemplos": int(v.sum()), "% del mínimo": 100 * v.sum() / base.sum(),
              "jugadoras": t.loc[v, "jugadora"].nunique()} for k, v in criterios.items()]
    return pd.DataFrame(filas).set_index("criterio")


def jugadoras_por_umbral(r: pd.DataFrame) -> pd.DataFrame:
    """Cuántas jugadoras y cuántos pares quedan según el mínimo de pares exigido."""
    total = r["pares_t_t1"].sum()
    filas = [{"minimo_pares": u, "jugadoras": int((r["pares_t_t1"] >= u).sum()),
              "pares": int(r.loc[r["pares_t_t1"] >= u, "pares_t_t1"].sum()),
              "% de pares": 100 * r.loc[r["pares_t_t1"] >= u, "pares_t_t1"].sum() / total}
             for u in (1, 30, 60, 100, 200)]
    return pd.DataFrame(filas).set_index("minimo_pares")


# ==============================================================================
def main() -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    TABLAS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)

    t = pd.read_csv(RAIZ / "data" / "processed" / "jugadora_dia.csv", parse_dates=["fecha"])
    # Readiness del día siguiente, SOLO para explorar. El calendario es completo,
    # así que shift(-1) dentro de cada jugadora es exactamente "mañana".
    t["readiness_sig"] = t.groupby("jugadora")["readiness"].shift(-1)

    # 1. Datos y huecos
    r = resumen_jugadoras(t)
    fig_dias_por_jugadora(r)
    fig_calendario(t, r)
    tasa = fig_respuesta_mensual(t)
    semana = 100 * t.groupby(t["fecha"].dt.dayofweek)["tiene_bienestar"].mean()
    semana.index = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]

    # 2. Distribución
    frec = fig_distribucion(t)
    fig_readiness_por_jugadora(r, t["readiness"].mean())

    # 3. Parecido con días anteriores
    ac = autocorrelacion(t)
    pares = t.dropna(subset=["readiness", "readiness_sig"])
    dif = (pares["readiness_sig"] - pares["readiness"]).clip(-3, 3).astype(int)
    cambio = 100 * dif.value_counts(normalize=True).sort_index()
    fig_autocorrelacion(ac, cambio)
    rachas = respuestas_repetidas(t)
    pers = persistencia_por_jugadora(pares)

    # 4. Relación con el objetivo
    co = correlaciones_objetivo(t)
    fig_correlaciones(co)
    matriz = fig_matriz_correlacion(t)
    elegidas = fig_series_ejemplo(t, r)

    # 5. Ejemplos disponibles
    ej = ejemplos_disponibles(t)
    umbral = jugadoras_por_umbral(r)

    # Guardar tablas
    r.to_csv(TABLAS / "resumen_jugadoras.csv", float_format="%.3f")
    ac.to_csv(TABLAS / "autocorrelacion.csv", float_format="%.3f")
    co.to_csv(TABLAS / "correlaciones_objetivo.csv", float_format="%.3f")
    matriz.to_csv(TABLAS / "matriz_correlacion.csv", float_format="%.3f")
    rachas.to_csv(TABLAS / "respuestas_repetidas.csv", float_format="%.1f")
    pers.to_csv(TABLAS / "persistencia_por_jugadora.csv", float_format="%.3f")
    ej.to_csv(TABLAS / "ejemplos_disponibles.csv", float_format="%.1f")
    umbral.to_csv(TABLAS / "jugadoras_por_umbral.csv", float_format="%.1f")

    # Resumen por pantalla
    print("=== 1. DATOS Y HUECOS ===")
    print(f"Cobertura sobre los 731 días: {100 * t['tiene_bienestar'].mean():.1f} %")
    print(f"Cobertura dentro del periodo activo de cada jugadora (mediana): "
          f"{r['cobertura_periodo_%'].median():.1f} %  (mín {r['cobertura_periodo_%'].min():.1f}, "
          f"máx {r['cobertura_periodo_%'].max():.1f})")
    print("Huecos por variable entre los días con cuestionario:")
    print((100 * t.loc[t["tiene_bienestar"] == 1, BIENESTAR].isna().mean()).round(2).to_string())
    print("Tasa de respuesta por día de la semana (%):", semana.round(1).to_dict())
    print("Tasa de respuesta mensual, mínimo y máximo (%):",
          tasa.min().round(1).to_dict(), tasa.max().round(1).to_dict())
    print("\n=== 2. DISTRIBUCIÓN ===")
    print(f"Media {t['readiness'].mean():.2f}, mediana {t['readiness'].median():.0f}, "
          f"desv. típica {t['readiness'].std():.2f}")
    print(f"Entre 5 y 9: {frec.loc[5:9].sum():.1f} %   |   3 o menos: {frec.loc[:3].sum():.1f} %")
    print(f"Media por jugadora: de {r['readiness_media'].min():.2f} a {r['readiness_media'].max():.2f}; "
          f"desv. típica entre jugadoras {r['readiness_media'].std():.2f}; "
          f"desv. típica media dentro de una jugadora {r['readiness_desv'].mean():.2f}")
    print("\n=== 3. PARECIDO CON DÍAS ANTERIORES ===")
    print(ac.loc[[1, 2, 3, 7, 14]].to_string())
    print("Cambio de hoy a mañana (%):", cambio.round(1).to_dict())
    cambio_max_1 = 100 * ((pares["readiness_sig"] - pares["readiness"]).abs() <= 1).mean()
    print(f"Cambia como mucho 1 punto: {cambio_max_1:.1f} %")
    print("Respuestas repetidas (rachas del mismo valor):")
    print(rachas.to_string())
    print(f"Días sin cambio por jugadora: de {pers['sin_cambio_pct'].min():.0f} % a "
          f"{pers['sin_cambio_pct'].max():.0f} %; MAE de persistencia por jugadora: de "
          f"{pers['mae_persistencia'].min():.2f} a {pers['mae_persistencia'].max():.2f}")
    print("\n=== 4. CORRELACIÓN CON LA READINESS DE MAÑANA ===")
    print(co.sort_values("intra_jugadora", ascending=False).to_string())
    print("Jugadoras de ejemplo:", elegidas)
    print("\n=== 5. EJEMPLOS DISPONIBLES ===")
    print(ej.to_string())
    print(umbral.to_string())


if __name__ == "__main__":
    main()
