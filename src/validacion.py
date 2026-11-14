"""
Piezas comunes de evaluación: las usan los baselines y, después, los modelos.

Lo importante de este fichero es que TODO lo que se compare (baselines y
modelos) use exactamente los mismos ejemplos, las mismas particiones y las
mismas métricas. Si no, la comparación no sería justa.

Contiene:
  - cargar_ejemplos(): los pares (día t -> readiness del día t+1)
  - particiones_por_jugadora(): validación cruzada sin mezclar jugadoras
  - particion_temporal(): entrenar con 2020 y evaluar en 2021
  - metricas(): MAE, RMSE y R2
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

RAIZ = Path(__file__).resolve().parent.parent
N_PARTICIONES = 5


def cargar_ejemplos() -> pd.DataFrame:
    """Devuelve una fila por ejemplo: un día t de una jugadora en el que se
    conoce su readiness y también la del día siguiente.

    La columna 'objetivo' es la readiness de t+1. El calendario de la tabla es
    completo (una fila por día), así que shift(-1) dentro de cada jugadora es
    exactamente 'mañana' y nunca 'el siguiente día con datos'.
    """
    t = pd.read_csv(RAIZ / "data" / "processed" / "jugadora_dia.csv", parse_dates=["fecha"])
    t = t.sort_values(["jugadora", "fecha"]).reset_index(drop=True)
    t["objetivo"] = t.groupby("jugadora")["readiness"].shift(-1)

    # Resúmenes del pasado de la propia jugadora. Los dos incluyen el día t y
    # ninguno mira más allá: se pueden conocer en el momento de predecir.
    por_jugadora = t.groupby("jugadora")["readiness"]
    #   media de todos sus cuestionarios hasta hoy (su "nivel habitual")
    t["media_propia"] = por_jugadora.transform(lambda x: x.expanding().mean())
    #   media de los últimos 7 días de calendario, con los que haya
    t["media_7d"] = por_jugadora.transform(lambda x: x.rolling(7, min_periods=1).mean())

    ejemplos = t.dropna(subset=["readiness", "objetivo"]).reset_index(drop=True)
    return ejemplos


def particiones_por_jugadora(ejemplos: pd.DataFrame):
    """Validación cruzada de 5 particiones agrupando por jugadora.

    Todas las filas de una jugadora caen en la misma partición, de modo que se
    evalúa siempre sobre jugadoras que NO se han visto al entrenar. Sin esto,
    el modelo podría 'reconocer' a la jugadora y el error saldría optimista.
    GroupKFold es determinista: reparte siempre igual, no hace falta semilla.

    Devuelve una lista de (nombre, índices de entrenamiento, índices de prueba).
    """
    gkf = GroupKFold(n_splits=N_PARTICIONES)
    return [
        (f"particion_{i + 1}", entrenamiento, prueba)
        for i, (entrenamiento, prueba) in enumerate(
            gkf.split(ejemplos, groups=ejemplos["jugadora"])
        )
    ]


def particion_temporal(ejemplos: pd.DataFrame):
    """Entrenar con la temporada 2020 y evaluar en la 2021.

    Aquí las jugadoras sí pueden repetirse: se comprueba otra cosa, que lo
    aprendido en una temporada sirve para la siguiente.
    """
    es_2020 = (ejemplos["temporada"] == 2020).to_numpy()
    return [("2020_a_2021", np.flatnonzero(es_2020), np.flatnonzero(~es_2020))]


def metricas(real: np.ndarray, prediccion: np.ndarray) -> dict:
    """MAE: error medio en puntos de readiness (fácil de interpretar).
    RMSE: como el MAE pero castiga más los errores grandes.
    R2: fracción de la variación explicada; 0 = igual que predecir la media
        de los datos evaluados, negativo = peor que eso."""
    error = np.asarray(real, dtype=float) - np.asarray(prediccion, dtype=float)
    varianza = ((real - np.mean(real)) ** 2).sum()
    return {
        "MAE": np.abs(error).mean(),
        "RMSE": np.sqrt((error ** 2).mean()),
        "R2": 1 - (error ** 2).sum() / varianza,
    }
