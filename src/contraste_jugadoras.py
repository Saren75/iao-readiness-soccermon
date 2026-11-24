"""
Contraste por jugadora (prueba de Wilcoxon).

POR QUÉ HACE FALTA
En Altair, el ANOVA compara los métodos mezclando los errores de todas las
jugadoras. Como unas jugadoras son mucho más fáciles de predecir que otras,
esa diferencia entre jugadoras es enorme y tapa la diferencia entre métodos:
el ANOVA no detecta nada.

La prueba de Wilcoxon es PAREADA: compara dos métodos jugadora a jugadora
(¿en esta jugadora, quién falla menos?) y así descuenta lo difícil que sea
cada una. No supone que los errores sigan una distribución normal.

QUÉ HACE
1. Repite la validación de Altair (proceso 04): una partición por jugadora,
   entrenando con las otras 49 y evaluando en ella.
2. Calcula el error (MAE y RMSE) de cada método en cada jugadora, con los
   modelos configurados como en Altair (modelo_altair.py).
3. Compara los métodos por parejas con Wilcoxon.

Sirve también para comprobar que Python y Altair dan los mismos errores, y
para calcular bien la persistencia, que en Altair estaba calculada de otra forma.

La red neuronal no se incluye: la de Altair usa un algoritmo de aprendizaje que
scikit-learn no tiene, así que no se puede reproducir de forma equivalente.

Uso (desde la carpeta del proyecto); tarda uno o dos minutos:
    .venv/bin/python src/contraste_jugadoras.py
"""

import numpy as np
import pandas as pd
from scipy import stats

from features import OBJETIVO, RAIZ, TODAS, cargar_ejemplos
from modelo_altair import random_forest_altair, ridge_altair

TABLAS = RAIZ / "reports" / "tablas"
METODOS = ["Persistencia", "Media de 7 días", "Ridge", "Random Forest"]


def errores_por_jugadora(ejemplos: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (jugadora, método) con su MAE y su RMSE."""
    filas = []
    for jugadora in sorted(ejemplos["jugadora"].unique()):
        prueba = ejemplos["jugadora"] == jugadora
        entrenamiento, test = ejemplos[~prueba], ejemplos[prueba]
        real = test[OBJETIVO].to_numpy()
        predicciones = {
            "Persistencia": test["readiness_hoy"].to_numpy(),
            "Media de 7 días": test["readiness_media_7d"].to_numpy(),
        }
        for nombre, crear in (("Ridge", ridge_altair), ("Random Forest", random_forest_altair)):
            modelo = crear().fit(entrenamiento[TODAS], entrenamiento[OBJETIVO])
            predicciones[nombre] = modelo.predict(test[TODAS])
        for metodo, pred in predicciones.items():
            error = real - pred
            filas.append({"jugadora": jugadora, "metodo": metodo, "n": len(test),
                          "MAE": np.abs(error).mean(), "RMSE": np.sqrt((error ** 2).mean())})
    return pd.DataFrame(filas)


def wilcoxon(errores: pd.DataFrame, metrica: str) -> pd.DataFrame:
    """Compara los métodos por parejas, jugadora a jugadora.
    p < 0,05 = la diferencia es estadísticamente significativa."""
    tabla = errores.pivot(index="jugadora", columns="metodo", values=metrica)
    pares = [("Random Forest", "Media de 7 días"), ("Ridge", "Media de 7 días"),
             ("Random Forest", "Persistencia"), ("Ridge", "Persistencia"),
             ("Media de 7 días", "Persistencia"), ("Random Forest", "Ridge")]
    filas = []
    for a, b in pares:
        filas.append({"comparacion": f"{a} vs {b}",
                      f"{metrica}_medio_a": tabla[a].mean(), f"{metrica}_medio_b": tabla[b].mean(),
                      "jugadoras_en_que_gana_a": int((tabla[a] < tabla[b]).sum()),
                      "p_wilcoxon": stats.wilcoxon(tabla[a], tabla[b]).pvalue})
    return pd.DataFrame(filas).set_index("comparacion")


def main() -> None:
    TABLAS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.float_format", "{:.3g}".format)

    errores = errores_por_jugadora(cargar_ejemplos())

    # Media entre jugadoras: cada jugadora cuenta igual, como en Altair
    resumen = errores.groupby("metodo")[["MAE", "RMSE"]].agg(["mean", "std"]).loc[METODOS]
    resumen.columns = [f"{m}_{'media' if e == 'mean' else 'desv'}" for m, e in resumen.columns]
    print("=== ERROR MEDIO POR JUGADORA (50 particiones, una por jugadora) ===")
    print(resumen.to_string())

    print("\n=== WILCOXON SOBRE EL RMSE DE CADA JUGADORA (criterio de Altair) ===")
    w_rmse = wilcoxon(errores, "RMSE")
    print(w_rmse.to_string())
    print("\n=== WILCOXON SOBRE EL MAE DE CADA JUGADORA ===")
    w_mae = wilcoxon(errores, "MAE")
    print(w_mae.to_string())

    errores.to_csv(TABLAS / "contraste_errores_por_jugadora.csv", index=False, float_format="%.4f")
    resumen.to_csv(TABLAS / "contraste_resumen.csv", float_format="%.4f")
    w_rmse.to_csv(TABLAS / "contraste_wilcoxon_rmse.csv", float_format="%.4g")
    w_mae.to_csv(TABLAS / "contraste_wilcoxon_mae.csv", float_format="%.4g")


if __name__ == "__main__":
    main()
