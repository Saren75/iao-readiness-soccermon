"""
Los modelos de Altair AI Studio, configurados igual en Python.

Los modelos del proyecto se entrenan y evalúan en Altair (procesos 04 y 05).
Este fichero define en Python modelos con la misma configuración, para poder
hacer lo que Altair no hace:
  - SHAP (explicabilidad.py), que tiene que explicar el mismo modelo.
  - La prueba de Wilcoxon por jugadora (contraste_jugadoras.py).

No son idénticos a los de Altair, porque cada herramienta programa los
algoritmos a su manera, pero sí equivalentes: mismos ajustes, mismo relleno de
huecos y mismo escalado. contraste_jugadoras.py comprueba que dan los mismos
errores que Altair.

Configuración de Altair (proceso 04):
  - Replace Missing Values con la media y Normalize (Z-transformation),
    ajustados solo con los datos de entrenamiento.
  - Linear Regression con ridge = 10.
  - Random Forest: 100 árboles, profundidad máxima 10, hojas de al menos 20
    ejemplos, todas las características en cada división, semilla 1992.
"""

import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# En macOS numpy lanza avisos falsos al multiplicar matrices ("encountered in
# matmul"); se comprobó que los resultados son correctos.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

SEMILLA = 1992  # la misma que en los procesos de Altair


def ridge_altair():
    """Linear Regression de Altair con ridge = 10."""
    return make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), Ridge(alpha=10.0))


def random_forest_altair():
    """Random Forest de Altair (100 árboles, profundidad 10, hojas de 20).
    Los árboles no necesitan escalado, así que solo se rellenan los huecos."""
    return make_pipeline(
        SimpleImputer(strategy="mean"),
        RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=20,
                              min_samples_split=4, max_features=1.0,
                              random_state=SEMILLA, n_jobs=-1),
    )
