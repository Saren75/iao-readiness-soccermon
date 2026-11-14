"""
Fase 3 - Objetivo y características.

Convierte la tabla jugadora-día en un conjunto de EJEMPLOS para aprendizaje
supervisado: cada fila es un día t de una jugadora, con lo que se sabía ese día
(entradas) y la readiness que declaró al día siguiente (objetivo).

Regla que cumplen todas las características: solo usan información de días
menores o iguales que t. Nada del día t+1 entra como entrada.

Uso (desde la carpeta "Practica final"):
    .venv/bin/python src/features.py

Salida: data/processed/ejemplos.csv (sirve también para Altair AI Studio).

Sufijos de las columnas creadas aquí:
    _d1, _d2   valor de ayer y de anteayer (desfases de 1 y 2 días)
    _m3, _m7   media de los últimos 3 y 7 días de calendario, contando hoy,
               calculada con los días que tengan dato

Decisiones (hay que saber explicarlas):
  1. No se descarta ningún ejemplo por tener huecos en los días anteriores.
     Los desfases que falten se dejan vacíos (NaN) y se rellenan DENTRO de la
     validación, usando solo los datos de entrenamiento. Además se añaden
     columnas que dicen si había dato (hay_ayer, hay_anteayer), para que el
     modelo pueda distinguir un valor real de uno rellenado.
  2. Las medias móviles se calculan con los días disponibles. Como el día t
     siempre tiene cuestionario, nunca quedan vacías.
  3. De las métricas de carga del dataset se quitan las duplicadas:
     carga_semanal (es ATL x 7), ctl42 (casi igual que ctl28) y duracion_min
     (casi igual que carga_diaria).
"""

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent

BIENESTAR = ["fatiga", "animo", "sueno_horas", "sueno_calidad", "dolor_muscular", "estres"]
DIAS = ["lun", "mar", "mie", "jue", "vie", "sab", "dom"]

# --- Grupos de características: se usan para comparar qué aporta cada bloque --
# 1) Historial de la propia readiness
G_READINESS = ["readiness", "readiness_d1", "readiness_d2", "readiness_m3",
               "readiness_m7", "media_propia"]
# 2) Contexto: cuántos datos hay detrás y qué día de la semana es mañana
G_CONTEXTO = (["hay_ayer", "hay_anteayer", "dias_con_datos_7d", "n_cuestionarios_previos"]
              + [f"manana_{d}" for d in DIAS])
# 3) Resto del cuestionario de bienestar
G_BIENESTAR = [f"{v}{s}" for v in BIENESTAR for s in ("", "_d1", "_d2", "_m3", "_m7")]
# 4) Carga de entrenamiento
G_CARGA = ["carga_diaria", "carga_diaria_d1", "carga_diaria_d2", "carga_diaria_m3",
           "atl", "ctl28", "acwr", "monotonia", "strain"]

TODAS = G_READINESS + G_CONTEXTO + G_BIENESTAR + G_CARGA


def construir_ejemplos() -> pd.DataFrame:
    t = pd.read_csv(RAIZ / "data" / "processed" / "jugadora_dia.csv", parse_dates=["fecha"])
    # El orden importa: todos los desplazamientos son por posición dentro de
    # cada jugadora, y la tabla tiene una fila por día de calendario, así que
    # "una fila atrás" es exactamente "ayer".
    t = t.sort_values(["jugadora", "fecha"]).reset_index(drop=True)
    g = t.groupby("jugadora")

    # --- Objetivo: la readiness de mañana --------------------------------------
    t["objetivo"] = g["readiness"].shift(-1)

    # --- Desfases y medias móviles del cuestionario ----------------------------
    for v in ["readiness"] + BIENESTAR:
        t[f"{v}_d1"] = g[v].shift(1)
        t[f"{v}_d2"] = g[v].shift(2)
        t[f"{v}_m3"] = g[v].transform(lambda x: x.rolling(3, min_periods=1).mean())
        t[f"{v}_m7"] = g[v].transform(lambda x: x.rolling(7, min_periods=1).mean())

    # --- Carga: desfases y media de 3 días (la de 7 días ya es el ATL) ---------
    t["carga_diaria_d1"] = g["carga_diaria"].shift(1)
    t["carga_diaria_d2"] = g["carga_diaria"].shift(2)
    t["carga_diaria_m3"] = g["carga_diaria"].transform(lambda x: x.rolling(3, min_periods=1).mean())

    # --- Nivel habitual de la jugadora: media de sus cuestionarios hasta hoy ---
    t["media_propia"] = g["readiness"].transform(lambda x: x.expanding().mean())

    # --- Contexto ---------------------------------------------------------------
    t["hay_ayer"] = g["tiene_bienestar"].shift(1).fillna(0).astype(int)
    t["hay_anteayer"] = g["tiene_bienestar"].shift(2).fillna(0).astype(int)
    t["dias_con_datos_7d"] = g["tiene_bienestar"].transform(lambda x: x.rolling(7, min_periods=1).sum())
    t["n_cuestionarios_previos"] = g["tiene_bienestar"].cumsum()
    # Día de la semana de MAÑANA (se conoce de antemano, no es información futura)
    manana = (t["fecha"] + pd.Timedelta(days=1)).dt.dayofweek
    for i, d in enumerate(DIAS):
        t[f"manana_{d}"] = (manana == i).astype(int)

    # --- Quedarse con los días que sirven de ejemplo ----------------------------
    # Hace falta la readiness de hoy (entrada principal) y la de mañana (objetivo).
    ejemplos = t.dropna(subset=["readiness", "objetivo"]).reset_index(drop=True)
    columnas = ["jugadora", "equipo", "temporada", "fecha"] + TODAS + ["objetivo"]
    return ejemplos[columnas]


def main() -> None:
    ejemplos = construir_ejemplos()
    ejemplos.to_csv(RAIZ / "data" / "processed" / "ejemplos.csv", index=False,
                    date_format="%Y-%m-%d", float_format="%.4f")
    print(f"Ejemplos: {len(ejemplos)} de {ejemplos['jugadora'].nunique()} jugadoras")
    print(f"Características: {len(TODAS)} "
          f"(readiness {len(G_READINESS)}, contexto {len(G_CONTEXTO)}, "
          f"bienestar {len(G_BIENESTAR)}, carga {len(G_CARGA)})")
    vacios = (100 * ejemplos[TODAS].isna().mean()).sort_values(ascending=False)
    print("Características con valores vacíos (% de ejemplos):")
    print(vacios[vacios > 0].round(1).to_string())


if __name__ == "__main__":
    main()
