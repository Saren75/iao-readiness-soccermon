"""
Fase 3 - Objetivo y características.

Convierte la tabla jugadora-día en un conjunto de ejemplos para aprendizaje
supervisado: cada fila es un día t de una jugadora (con cuestionario en t y en
t+1), con la información disponible hasta ese día (entradas) y la readiness
declarada al día siguiente (objetivo, columna readiness_manana).

Se usa de dos formas, con el mismo código:
  - Desde la terminal (carpeta del proyecto):  .venv/bin/python src/features.py
    Lee data/processed/jugadora_dia.csv y guarda data/processed/ejemplos.csv.
  - Desde Altair AI Studio: el proceso 03_Ingenieria_Caracteristicas lo ejecuta
    con el operador Execute Python, que llama a rm_main() con la tabla.

Regla anti-fuga: las entradas solo usan información observada en días menores o
iguales que t (la única referencia a t+1 es el día de la semana de mañana,
conocido de antemano).

56 características en cuatro bloques:
- Historial de readiness (6): hoy, ayer, anteayer, medias de 3 y 7 días, y media
  propia acumulada hasta t.
- Contexto (11): si hubo cuestionario ayer y anteayer, días con datos en los
  últimos 7 días, cuestionarios acumulados y día de la semana de t+1.
- Bienestar (30): hoy, ayer, anteayer y medias de 3 y 7 días para las otras 6
  variables del cuestionario (fatiga, ánimo, horas y calidad de sueño, dolor
  muscular y estrés).
- Carga (9): carga diaria (hoy, ayer, anteayer, media de 3 días), ATL, CTL de
  28 días, ACWR, monotonía y strain.

Decisiones:
1. No se descarta ningún ejemplo por tener huecos en los días anteriores. Los
   desfases que falten se dejan vacíos y se rellenan DENTRO de la validación,
   con un relleno ajustado solo con los datos de entrenamiento (en Altair,
   Replace Missing Values con la media). Las columnas tiene_bienestar_ayer y
   tiene_bienestar_anteayer distinguen un dato real de uno rellenado.
2. Las medias móviles se calculan con los días disponibles. Como el día t
   siempre tiene cuestionario, nunca quedan vacías.
3. De las métricas de carga del dataset se eliminan las redundantes:
   carga_semanal (es ATL x 7), ctl42 (muy correlacionada con ctl28) y los
   minutos entrenados (muy correlacionados con la carga diaria).
"""

from pathlib import Path

import pandas as pd

VARS_BIENESTAR = ["fatiga", "animo", "sueno_horas", "sueno_calidad", "dolor_muscular", "estres"]

# --- Bloques de características (los usan los scripts de Python) -------------
G_READINESS = ["readiness_hoy", "readiness_ayer", "readiness_anteayer",
               "readiness_media_3d", "readiness_media_7d", "readiness_media_propia"]
G_CONTEXTO = (["tiene_bienestar_ayer", "tiene_bienestar_anteayer", "dias_datos_7d",
               "cuestionarios_acumulados"] + [f"dia_manana_{d}" for d in range(7)])
G_BIENESTAR = [f"{v}_{s}" for v in VARS_BIENESTAR
               for s in ("hoy", "ayer", "anteayer", "media_3d", "media_7d")]
G_CARGA = ["carga_hoy", "carga_ayer", "carga_anteayer", "carga_media_3d",
           "atl", "ctl28", "acwr", "monotonia", "strain"]
TODAS = G_READINESS + G_CONTEXTO + G_BIENESTAR + G_CARGA
OBJETIVO = "readiness_manana"

# Dentro de Altair no existe __file__
try:
    RAIZ = Path(__file__).resolve().parent.parent
except NameError:
    RAIZ = None


def construir_ejemplos(tabla: pd.DataFrame) -> pd.DataFrame:
    df = tabla.copy()
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.tz_localize(None)
    # El orden importa: todos los desplazamientos son por posición dentro de
    # cada jugadora, y la tabla tiene una fila por día de calendario, así que
    # "una fila atrás" es exactamente "ayer".
    df = df.sort_values(["jugadora", "fecha"]).reset_index(drop=True)
    g = df.groupby("jugadora", sort=False)

    def media_movil(columna, dias):
        return g[columna].transform(lambda s: s.rolling(dias, min_periods=1).mean())

    # Objetivo: readiness del día siguiente (t + 1)
    df[OBJETIVO] = g["readiness"].shift(-1)

    feats = pd.DataFrame(index=df.index)

    # 1. Historial de readiness (6)
    feats["readiness_hoy"] = df["readiness"]
    feats["readiness_ayer"] = g["readiness"].shift(1)
    feats["readiness_anteayer"] = g["readiness"].shift(2)
    feats["readiness_media_3d"] = media_movil("readiness", 3)
    feats["readiness_media_7d"] = media_movil("readiness", 7)
    # Nivel habitual de la jugadora: media de todos sus cuestionarios hasta hoy
    feats["readiness_media_propia"] = g["readiness"].transform(
        lambda s: s.expanding(min_periods=1).mean())

    # 2. Contexto (11)
    feats["tiene_bienestar_ayer"] = g["tiene_bienestar"].shift(1).fillna(0).astype(int)
    feats["tiene_bienestar_anteayer"] = g["tiene_bienestar"].shift(2).fillna(0).astype(int)
    feats["dias_datos_7d"] = g["tiene_bienestar"].transform(
        lambda s: s.rolling(7, min_periods=1).sum()).astype(int)
    feats["cuestionarios_acumulados"] = g["tiene_bienestar"].cumsum().astype(int)
    # Día de la semana de MAÑANA (0 = lunes): se conoce de antemano
    dia_manana = (df["fecha"] + pd.Timedelta(days=1)).dt.dayofweek
    for d in range(7):
        feats[f"dia_manana_{d}"] = (dia_manana == d).astype(int)

    # 3. Bienestar (30: 6 variables x 5)
    for col in VARS_BIENESTAR:
        feats[f"{col}_hoy"] = df[col]
        feats[f"{col}_ayer"] = g[col].shift(1)
        feats[f"{col}_anteayer"] = g[col].shift(2)
        feats[f"{col}_media_3d"] = media_movil(col, 3)
        feats[f"{col}_media_7d"] = media_movil(col, 7)

    # 4. Carga (9, sin las redundantes)
    feats["carga_hoy"] = df["carga_diaria"]
    feats["carga_ayer"] = g["carga_diaria"].shift(1)
    feats["carga_anteayer"] = g["carga_diaria"].shift(2)
    feats["carga_media_3d"] = media_movil("carga_diaria", 3)
    for col in ["atl", "ctl28", "acwr", "monotonia", "strain"]:
        feats[col] = df[col]

    # Ensamblar metadatos + 56 características + objetivo
    meta = df[["jugadora", "equipo", "temporada", "fecha", OBJETIVO]]
    ejemplos = pd.concat([meta, feats[TODAS]], axis=1)

    # Solo pares válidos: readiness conocida hoy (t) y mañana (t+1)
    validos = ejemplos["readiness_hoy"].notna() & ejemplos[OBJETIVO].notna()
    return ejemplos.loc[validos].reset_index(drop=True)


def rm_main(tabla):
    """Punto de entrada para Altair AI Studio (operador Execute Python)."""
    return construir_ejemplos(tabla)


def cargar_ejemplos() -> pd.DataFrame:
    """Construye los ejemplos a partir de la tabla jugadora-día guardada."""
    tabla = pd.read_csv(RAIZ / "data" / "processed" / "jugadora_dia.csv", parse_dates=["fecha"])
    return construir_ejemplos(tabla)


def main() -> None:
    """Punto de entrada desde la terminal: guarda los ejemplos en CSV."""
    ejemplos = cargar_ejemplos()
    ejemplos.to_csv(RAIZ / "data" / "processed" / "ejemplos.csv", index=False,
                    date_format="%Y-%m-%d", float_format="%.4f")
    print(f"Ejemplos: {len(ejemplos)} de {ejemplos['jugadora'].nunique()} jugadoras "
          f"(2020: {(ejemplos['temporada'] == 2020).sum()}, 2021: {(ejemplos['temporada'] == 2021).sum()})")
    print(f"Características: {len(TODAS)} (readiness {len(G_READINESS)}, contexto {len(G_CONTEXTO)}, "
          f"bienestar {len(G_BIENESTAR)}, carga {len(G_CARGA)})")


if __name__ == "__main__":
    main()
