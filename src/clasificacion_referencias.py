"""
AUC de las dos reglas sencillas de la clasificación de días malos.

La clasificación se hace en Altair (proceso 07_Clasificacion_Dia_Malo). Allí
las reglas «mañana malo si hoy fue malo» (readiness de hoy <= 4) y «si la media
de 7 días es 4 o menos» se evalúan con recall, precisión y F1. Pero una regla
fija solo dice sí o no, y Altair no deja darle una puntuación, así que no
calcula su AUC.

Aquí se calcula ese AUC ordenando los días por la propia readiness: cuanto más
baja la readiness de hoy (o la media de 7 días), más riesgo de día malo. No se
entrena nada, así que no hace falta validación cruzada: el resultado es el
mismo con cualquier partición.

Uso (desde la carpeta del proyecto):
    .venv/bin/python src/clasificacion_referencias.py
"""

import pandas as pd
from sklearn.metrics import roc_auc_score

from features import OBJETIVO, RAIZ, cargar_ejemplos

TABLAS = RAIZ / "reports" / "tablas"
UMBRAL = 4  # día malo: readiness de mañana de 4 o menos


def main() -> None:
    ejemplos = cargar_ejemplos()
    dia_malo = (ejemplos[OBJETIVO] <= UMBRAL).astype(int)
    print(f"Días malos: {dia_malo.sum()} de {len(dia_malo)} ({100 * dia_malo.mean():.1f} %)")

    filas = []
    for nombre, columna in (("Persistencia (hoy <= 4)", "readiness_hoy"),
                            ("Media de 7 días <= 4", "readiness_media_7d")):
        # Se usa la readiness con signo negativo: a menos readiness, más riesgo
        filas.append({"regla": nombre, "AUC": roc_auc_score(dia_malo, -ejemplos[columna])})
    tabla = pd.DataFrame(filas).set_index("regla")
    print(tabla.round(3).to_string())
    tabla.to_csv(TABLAS / "clasificacion_auc_referencias.csv", float_format="%.4f")


if __name__ == "__main__":
    main()
