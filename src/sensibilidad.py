"""
Análisis de sensibilidad: tres aspectos de los datos que podrían cambiar las conclusiones.

Se repite la validación por jugadoras de Altair (una partición por jugadora,
modelos configurados como en Altair) cambiando una sola cosa cada vez, y se
mira si cambian las conclusiones:

1. NOTAS REPETIDAS. Algunas jugadoras ponen la misma readiness durante semanas.
   En esos tramos «mañana igual que hoy» acierta siempre sin que eso sea
   capacidad de predicción. Variante: quitar los días que forman parte de una
   racha de 14 o más días seguidos con la misma nota (como día de hoy o como
   día de mañana), tanto del entrenamiento como de la evaluación.

2. CEROS DE CARGA. Una carga de 0 no distingue «no entrenó» de «no lo
   registró». Variante: añadir qué parte del resto del equipo registró una
   sesión ese día (equipo_entreno_hoy, de 0 a 1). Si el equipo entrenó y ella
   tiene 0, probablemente no entrenó (descanso, lesión) o no lo apuntó; si
   nadie entrenó, el 0 es un día libre del equipo.

3. VARIABLES REDUNDANTES. El strain está muy correlacionado con la carga
   semanal y la monotonía. Variante: quitar el strain.

Se compara cada variante con la configuración base en las mismas jugadoras, y
se repite la prueba de Wilcoxon de los modelos frente a la media de 7 días.

Uso (desde la carpeta del proyecto); tarda unos cinco minutos:
    .venv/bin/python src/sensibilidad.py
"""

import numpy as np
import pandas as pd
from scipy import stats

from features import OBJETIVO, RAIZ, TODAS, cargar_ejemplos
from modelo_altair import random_forest_altair, ridge_altair

TABLAS = RAIZ / "reports" / "tablas"
RACHA_MINIMA = 14
METODOS = ["Persistencia", "Media de 7 días", "Ridge", "Random Forest"]


def dias_en_racha(tabla: pd.DataFrame, minimo: int = RACHA_MINIMA) -> pd.DataFrame:
    """(jugadora, fecha) de los cuestionarios que forman parte de una racha de
    `minimo` o más días seguidos con la misma readiness (misma definición que
    el análisis exploratorio)."""
    c = tabla[tabla["tiene_bienestar"] == 1].sort_values(["jugadora", "fecha"]).copy()
    g = c.groupby("jugadora")
    nueva = g["readiness"].diff().ne(0) | g["fecha"].diff().dt.days.ne(1)
    c["racha"] = nueva.cumsum()
    c["largo"] = c.groupby("racha")["readiness"].transform("size")
    return c.loc[c["largo"] >= minimo, ["jugadora", "fecha"]]


def equipo_entreno(tabla: pd.DataFrame) -> pd.DataFrame:
    """Parte del RESTO del equipo (sin contar a la jugadora) que registró al
    menos una sesión ese día."""
    t = tabla[["jugadora", "equipo", "fecha", "n_sesiones"]].copy()
    t["entreno"] = (t["n_sesiones"] > 0).astype(int)
    grupo = t.groupby(["equipo", "fecha"])["entreno"]
    total, n = grupo.transform("sum"), grupo.transform("size")
    t["equipo_entreno_hoy"] = (total - t["entreno"]) / (n - 1)
    return t[["jugadora", "fecha", "equipo_entreno_hoy"]]


def validar(ejemplos: pd.DataFrame, entradas: list[str]) -> pd.DataFrame:
    """Validación por jugadoras. Devuelve MAE y RMSE por (jugadora, método)."""
    filas = []
    for jugadora in sorted(ejemplos["jugadora"].unique()):
        prueba = ejemplos["jugadora"] == jugadora
        entrenamiento, test = ejemplos[~prueba], ejemplos[prueba]
        real = test[OBJETIVO].to_numpy()
        predicciones = {"Persistencia": test["readiness_hoy"].to_numpy(),
                        "Media de 7 días": test["readiness_media_7d"].to_numpy()}
        for nombre, crear in (("Ridge", ridge_altair), ("Random Forest", random_forest_altair)):
            modelo = crear().fit(entrenamiento[entradas], entrenamiento[OBJETIVO])
            predicciones[nombre] = modelo.predict(test[entradas])
        for metodo, pred in predicciones.items():
            error = real - pred
            filas.append({"jugadora": jugadora, "metodo": metodo,
                          "MAE": np.abs(error).mean(), "RMSE": np.sqrt((error ** 2).mean())})
    return pd.DataFrame(filas)


def resumir(nombre: str, errores: pd.DataFrame, n_ejemplos: int) -> dict:
    """Media por jugadora de cada método y Wilcoxon de los modelos frente a la
    media de 7 días."""
    fila = {"variante": nombre, "ejemplos": n_ejemplos, "jugadoras": errores["jugadora"].nunique()}
    for metrica in ("RMSE", "MAE"):
        tabla = errores.pivot(index="jugadora", columns="metodo", values=metrica)
        for m in METODOS:
            fila[f"{metrica} {m}"] = tabla[m].mean()
        for m in ("Ridge", "Random Forest"):
            fila[f"{metrica} p {m} vs 7 días"] = stats.wilcoxon(tabla[m], tabla["Media de 7 días"]).pvalue
    return fila


def main() -> None:
    pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format)
    tabla = pd.read_csv(RAIZ / "data" / "processed" / "jugadora_dia.csv", parse_dates=["fecha"])
    tabla["fecha"] = tabla["fecha"].dt.tz_localize(None)
    ejemplos = cargar_ejemplos()

    # Variante 2: añadir equipo_entreno_hoy
    ejemplos = ejemplos.merge(equipo_entreno(tabla), on=["jugadora", "fecha"],
                              how="left", validate="one_to_one")
    # Variante 1: marcar los ejemplos cuyo día de hoy o de mañana está en una racha larga
    racha = dias_en_racha(tabla).assign(en_racha=1)
    hoy = ejemplos.merge(racha, on=["jugadora", "fecha"], how="left")["en_racha"].fillna(0)
    manana = ejemplos.assign(fecha=ejemplos["fecha"] + pd.Timedelta(days=1)).merge(
        racha, on=["jugadora", "fecha"], how="left")["en_racha"].fillna(0)
    en_racha = (hoy + manana > 0).to_numpy()
    print(f"Ejemplos en rachas de {RACHA_MINIMA}+ días: {en_racha.sum()} de {len(ejemplos)}")

    # Cuánto ayuda la nueva variable a distinguir ceros
    ceros = ejemplos["carga_hoy"] == 0
    print("Días con carga 0: el resto del equipo entrenó (> 50 %) en "
          f"{100 * (ejemplos.loc[ceros, 'equipo_entreno_hoy'] > 0.5).mean():.0f} % de ellos")

    variantes = [
        ("Base (56 características)", ejemplos, TODAS),
        (f"Sin rachas de {RACHA_MINIMA}+ días", ejemplos[~en_racha], TODAS),
        ("Con equipo_entreno_hoy", ejemplos, TODAS + ["equipo_entreno_hoy"]),
        ("Sin strain", ejemplos, [c for c in TODAS if c != "strain"]),
    ]
    filas, por_jugadora = [], {}
    for nombre, datos, entradas in variantes:
        print(f"  {nombre}...")
        errores = validar(datos, entradas)
        por_jugadora[nombre] = errores
        filas.append(resumir(nombre, errores, len(datos)))
    resumen = pd.DataFrame(filas).set_index("variante")

    # Cada variante frente a la base, jugadora a jugadora (Random Forest).
    # La variante sin rachas cambia los días evaluados, así que no es pareable.
    base = por_jugadora[variantes[0][0]].query("metodo == 'Random Forest'").set_index("jugadora")["RMSE"]
    for nombre in ("Con equipo_entreno_hoy", "Sin strain"):
        otra = por_jugadora[nombre].query("metodo == 'Random Forest'").set_index("jugadora")["RMSE"]
        resumen.loc[nombre, "p RF frente a la base (RMSE)"] = stats.wilcoxon(otra, base).pvalue

    print("\n=== RMSE MEDIO POR JUGADORA ===")
    print(resumen[["ejemplos", "jugadoras"] + [f"RMSE {m}" for m in METODOS]].to_string())
    print("\n=== MAE MEDIO POR JUGADORA ===")
    print(resumen[[f"MAE {m}" for m in METODOS]].to_string())
    print("\n=== WILCOXON (p) ===")
    print(resumen[[c for c in resumen.columns if " p " in f" {c}"]].to_string(float_format="{:.2g}".format))

    resumen.to_csv(TABLAS / "sensibilidad_resumen.csv", float_format="%.4g")


if __name__ == "__main__":
    main()
