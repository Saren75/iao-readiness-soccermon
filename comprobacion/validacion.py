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

from features import construir_ejemplos

RAIZ = Path(__file__).resolve().parent.parent
N_PARTICIONES = 5


def cargar_ejemplos() -> pd.DataFrame:
    """Devuelve los ejemplos (día t -> readiness de t+1) con sus características.
    Se construyen en features.py; aquí solo se reexportan para que baselines y
    modelos partan exactamente de las mismas filas."""
    return construir_ejemplos()


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
