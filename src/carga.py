"""
Fase 1 - Carga de SoccerMon (parte subjetiva) y construcción de la tabla base.

Lee los ficheros de data/raw/subjective/ y genera UNA tabla con una fila por
jugadora y día (formato "largo"), que es lo que necesitan tanto Python como
Altair AI Studio para trabajar.

Se usa de dos formas, con el mismo código:
  - Desde la terminal (carpeta del proyecto):  .venv/bin/python src/carga.py
  - Desde Altair AI Studio: el proceso 01_Ingesta_y_Limpieza lo ejecuta con el
    operador Execute Python, que llama a rm_main() con la macro repo_dir.

Salidas (en data/processed/):
    jugadora_dia.csv   -> la tabla base (50 jugadoras x 731 días = 36.550 filas)
    jugadoras.csv      -> correspondencia entre el código corto (A01, B07...) y
                          el identificador original del dataset

Decisiones:
  1. Se conserva el calendario COMPLETO (todos los días de 2020 y 2021 para cada
     jugadora), aunque no haya datos. Así "la fila anterior" es siempre "ayer",
     y los desfases y ventanas de la Fase 3 no se equivocan con los huecos.
  2. Día sin cuestionario de bienestar -> NaN. No se inventa ni se rellena nada.
  3. La carga viene ya sin huecos en el dataset: los días sin sesión valen 0.
     Un 0 no distingue "no entrenó" de "no lo registró" (limitación).
  4. No se crea todavía ninguna característica nueva ni el objetivo: esta tabla
     solo reorganiza lo que ya está medido en el dataset.
"""

import json
from pathlib import Path

import pandas as pd

# --- Rutas (relativas a la carpeta del proyecto, no a donde se ejecute) -------
# Dentro de Altair no existe __file__: allí la carpeta llega en la macro repo_dir.
try:
    RAIZ = Path(__file__).resolve().parent.parent
except NameError:
    RAIZ = None

# --- Variables del dataset: nombre del fichero -> nombre de la columna --------
# Bienestar: cuestionario diario. readiness va de 1 a 10; sleep_duration son
# horas; el resto va de 1 a 5.
BIENESTAR = {
    "readiness": "readiness",
    "fatigue": "fatiga",
    "mood": "animo",
    "sleep_duration": "sueno_horas",
    "sleep_quality": "sueno_calidad",
    "soreness": "dolor_muscular",
    "stress": "estres",
}

# Carga: métricas que el dataset ya trae calculadas a partir de las sesiones.
# Todas se calculan con días <= t (comprobado contra daily_load):
#   carga_diaria   = suma de sRPE (RPE x duración) de las sesiones del día
#   carga_semanal  = suma de carga_diaria de los últimos 7 días
#   atl            = media de los últimos 7 días   (carga aguda)
#   ctl28 / ctl42  = media de los últimos 28 / 42 días (carga crónica)
#   acwr           = atl / ctl28 (relación aguda:crónica)
#   monotonia      = media / desviación típica de los últimos 7 días
#   strain         = carga_semanal x monotonia (definición habitual; al
#                    recalcularla sale parecida pero no idéntica a la del dataset)
CARGA = {
    "daily_load": "carga_diaria",
    "weekly_load": "carga_semanal",
    "atl": "atl",
    "ctl28": "ctl28",
    "ctl42": "ctl42",
    "acwr": "acwr",
    "monotony": "monotonia",
    "strain": "strain",
}

CLAVE = ["jugadora_id", "fecha"]


def leer_ancho(ruta: Path, nombre: str) -> pd.DataFrame:
    """Lee un CSV 'ancho' (filas = días, columnas = jugadoras) y lo pasa a
    'largo': una fila por (jugadora, fecha) con una sola columna de valor."""
    df = pd.read_csv(ruta)
    # La primera columna es la fecha, pero su cabecera cambia de un fichero a
    # otro ("Readiness Data", "Date"...), así que se renombra por posición.
    df = df.rename(columns={df.columns[0]: "fecha"})
    df["fecha"] = pd.to_datetime(df["fecha"], format="%d.%m.%Y")
    largo = df.melt(id_vars="fecha", var_name="jugadora_id", value_name=nombre)
    largo[nombre] = pd.to_numeric(largo[nombre], errors="coerce")
    return largo


def leer_sesiones(ruta: Path) -> pd.DataFrame:
    """Lee session.json (una entrada por sesión de entrenamiento) y lo resume
    por jugadora y día: nº de sesiones, minutos totales y RPE medio."""
    with open(ruta, encoding="utf-8") as f:
        crudo = json.load(f)
    filas = [
        {"jugadora_id": jugadora, **sesion}
        for jugadora, sesiones in crudo.items()
        for sesion in sesiones
    ]
    ses = pd.DataFrame(filas)
    ses["fecha"] = pd.to_datetime(ses["date"], format="%d.%m.%Y")
    return (
        ses.groupby(CLAVE)
        .agg(
            n_sesiones=("srpe", "size"),
            duracion_min=("duration", "sum"),
            rpe_medio=("rpe", "mean"),
        )
        .reset_index()
    )


def construir_tabla(raiz: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (tabla jugadora-día, tabla de correspondencia de jugadoras)."""
    crudos = raiz / "data" / "raw" / "subjective"
    # 1) Leer cada variable y unirlas por (jugadora, fecha). Todos los ficheros
    #    tienen las mismas 50 jugadoras y los mismos 731 días, así que la unión
    #    no pierde ni duplica filas (se comprueba más abajo).
    tabla = None
    for carpeta, variables in (("wellness", BIENESTAR), ("training-load", CARGA)):
        for fichero, nombre in variables.items():
            parte = leer_ancho(crudos / carpeta / f"{fichero}.csv", nombre)
            tabla = parte if tabla is None else tabla.merge(
                parte, on=CLAVE, how="outer", validate="one_to_one"
            )

    # 2) Añadir el resumen de sesiones. Día sin sesión -> 0 sesiones y 0 minutos;
    #    el RPE medio se deja en NaN porque no hay nada que promediar.
    tabla = tabla.merge(
        leer_sesiones(crudos / "training-load" / "session.json"),
        on=CLAVE, how="left", validate="one_to_one",
    )
    tabla["n_sesiones"] = tabla["n_sesiones"].fillna(0).astype(int)
    tabla["duracion_min"] = tabla["duracion_min"].fillna(0)

    # 3) Equipo y temporada. El equipo va dentro del identificador original
    #    ("TeamA-xxxx"); la temporada es el año natural.
    tabla["equipo"] = tabla["jugadora_id"].str.slice(0, 5).str.replace("Team", "")
    tabla["temporada"] = tabla["fecha"].dt.year

    # 4) Código corto y legible por jugadora (A01...A27, B01...B23), respetando
    #    el orden en que aparecen en el dataset. El original se guarda aparte.
    ids = pd.read_csv(crudos / "wellness" / "readiness.csv", nrows=0).columns[1:]
    jugadoras = pd.DataFrame({"jugadora_id": ids})
    jugadoras["equipo"] = jugadoras["jugadora_id"].str.slice(4, 5)
    numero = jugadoras.groupby("equipo").cumcount() + 1
    jugadoras["jugadora"] = jugadoras["equipo"] + numero.astype(str).str.zfill(2)
    tabla = tabla.merge(jugadoras[["jugadora_id", "jugadora"]], on="jugadora_id")

    # 5) Marcar si ese día hay cuestionario (basta con que haya readiness, que es
    #    la variable que se quiere predecir).
    tabla["tiene_bienestar"] = tabla["readiness"].notna().astype(int)

    # 6) Orden final: por jugadora y fecha, que es como se leerá siempre.
    columnas = (
        ["jugadora", "equipo", "temporada", "fecha", "tiene_bienestar"]
        + list(BIENESTAR.values())
        + ["n_sesiones", "duracion_min", "rpe_medio"]
        + list(CARGA.values())
    )
    tabla = tabla.sort_values(["jugadora", "fecha"])[columnas].reset_index(drop=True)
    return tabla, jugadoras[["jugadora", "equipo", "jugadora_id"]]


def comprobar(tabla: pd.DataFrame) -> None:
    """Comprobaciones de integridad: si algo falla, se para con un error."""
    n_jug = tabla["jugadora"].nunique()
    n_dias = tabla["fecha"].nunique()
    assert len(tabla) == n_jug * n_dias, "Faltan o sobran filas"
    assert not tabla.duplicated(["jugadora", "fecha"]).any(), "Filas duplicadas"
    # Calendario diario sin saltos dentro de cada jugadora
    saltos = tabla.groupby("jugadora")["fecha"].diff().dropna().dt.days
    assert (saltos == 1).all(), "El calendario tiene saltos"
    assert tabla["readiness"].dropna().between(1, 10).all(), "Readiness fuera de 1-10"
    assert tabla[list(CARGA.values())].notna().all().all(), "Carga con huecos"


def rm_main(macros):
    """Punto de entrada para Altair AI Studio (operador Execute Python).
    Devuelve las dos tablas; el proceso las guarda con Store."""
    tabla, jugadoras = construir_tabla(Path(macros["repo_dir"]))
    comprobar(tabla)
    return tabla, jugadoras


def main() -> None:
    """Punto de entrada desde la terminal: guarda las tablas en CSV."""
    tabla, jugadoras = construir_tabla(RAIZ)
    salida = RAIZ / "data" / "processed"
    comprobar(tabla)

    salida.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida / "jugadora_dia.csv", index=False, date_format="%Y-%m-%d")
    jugadoras.to_csv(salida / "jugadoras.csv", index=False)

    print(f"Tabla guardada: {len(tabla)} filas x {tabla.shape[1]} columnas")
    print(f"  Jugadoras: {tabla['jugadora'].nunique()} "
          f"(A: {(jugadoras['equipo'] == 'A').sum()}, B: {(jugadoras['equipo'] == 'B').sum()})")
    print(f"  Fechas: {tabla['fecha'].min():%Y-%m-%d} a {tabla['fecha'].max():%Y-%m-%d}")
    print(f"  Días con cuestionario de bienestar: {tabla['tiene_bienestar'].sum()} "
          f"({100 * tabla['tiene_bienestar'].mean():.1f} %)")
    print(f"  Días con al menos una sesión: {(tabla['n_sesiones'] > 0).sum()}")


if __name__ == "__main__":
    main()
