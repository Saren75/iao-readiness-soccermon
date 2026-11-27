# Guía de los procesos de Altair AI Studio

Cómo reproducir en Altair AI Studio la parte del proyecto que se hace allí (procesos `01` a `07` de `processes/`) y qué resultados debe dar cada proceso. El problema, los datos, la justificación de cada decisión y las conclusiones están en el [`README.md`](README.md).

## 1. Preparación

### Requisitos

* **Altair AI Studio** (probado con la versión 2026.1.1).
* Extensión **Python Scripting** (`Extensions → Marketplace → Python Scripting`), vinculada a un entorno de Python con `pandas` y `numpy` (`Settings → Preferences → Python Scripting`). Solo la usan los procesos 01 y 03; sirve el mismo entorno que se crea en el README con `requirements.txt`.

### Paso 1: abrir el proyecto como repositorio de Altair

Los procesos leen y guardan sus datos (`Retrieve` y `Store`) en el repositorio `//Grupo10_SoccerMon/...`:

1. Clona o descarga este repositorio en tu equipo.
2. En Altair AI Studio, en el panel **Repositories**, pulsa **`+` → New local repository**.
3. Llama al repositorio exactamente **`Grupo10_SoccerMon`** (los procesos lo buscan por ese nombre) y elige como *Root Directory* la carpeta raíz del proyecto.
4. Comprueba que aparecen las carpetas `data`, `models`, `processes`, `reports` y `src`.

### Paso 2: indicar la ruta del proyecto en los procesos 01 y 03

Los procesos `01_Ingesta_y_Limpieza` y `03_Ingenieria_Caracteristicas` ejecutan `src/carga.py` y `src/features.py` con `Execute Python`, y encuentran esos ficheros a través de la macro **`repo_dir`** (el parámetro *script file* vale `%{repo_dir}/src/carga.py` y `%{repo_dir}/src/features.py`; el 01 la usa también para leer `data/raw/subjective/`).

En cada uno de los dos procesos, abre el panel **Context** (`View → Show Panel → Context`) y cambia el valor de **`repo_dir`** por la ruta absoluta de la carpeta del proyecto en tu equipo, sin barra final. Si no se cambia, el proceso falla con *Execution of script failed* porque no encuentra el script.

El repositorio ya incluye las tablas procesadas y los modelos guardados, así que también se puede empezar directamente por el proceso 02 o el 04.

### Paso 3: ejecutar los procesos en orden

Cada proceso guarda lo que necesita el siguiente (`Store` → `Retrieve`):

| Orden | Proceso | Qué hace | Lee | Guarda | Tiempo aprox. |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 1 | `01_Ingesta_y_Limpieza` | Construye la tabla jugadora-día con `src/carga.py` | `data/raw/subjective/` | `data/processed/Jugadora_dia`, `data/processed/jugadoras` | segundos |
| 2 | `02_AED_Exploracion` | Vistas principales del análisis exploratorio | `Jugadora_dia` | Resultados en pantalla | segundos |
| 3 | `03_Ingenieria_Caracteristicas` | Construye los 14.631 ejemplos con 56 características con `src/features.py` y asigna los roles | `Jugadora_dia` | `data/processed/ejemplos` | segundos |
| 4 | `04_Baselines_y_Modelado` | Validación por jugadoras de las dos referencias y los tres modelos, ANOVA y T-Test | `ejemplos` | `models/modelo_ridge_cv`, `models/modelo_rf_cv`, `models/modelo_nn_cv` | ~10 min |
| 5 | `05_Validacion_Temporal_2020_2021` | Entrena con 2020 y evalúa en 2021 | `ejemplos` | `models/pipeline_rf_2020`, `models/pesos_rf_2020` | unos minutos |
| 6 | `06_Explicabilidad_y_Errores` | Importancia de variables y error por franja de readiness, sin reentrenar | `pipeline_rf_2020`, `pesos_rf_2020`, `ejemplos` | Resultados en pantalla | segundos |
| 7 | `07_Clasificacion_Dia_Malo` | Clasificación de días malos con la misma validación por jugadoras | `ejemplos` | `models/modelo_rf_clasificacion_cv` | unos minutos |

## 2. Cómo está construido cada proceso

Solo lo propio de Altair; el porqué de cada decisión está en el apartado 2 del README.

* **Roles (proceso 03).** `readiness_manana` es la etiqueta (`label`); `jugadora` tiene el rol `batch`; `equipo`, `temporada` y `fecha` son `metadata`, así que no entran como variables.
* **Validación por jugadoras (procesos 04 y 07).** `Cross Validation` con **`split on batch attribute`** activado: hay una partición por cada valor de `jugadora` (50) y cada una deja fuera a una jugadora entera. Con esta opción, el número de particiones del operador no se usa.
* **Preparación dentro de la validación (procesos 04, 05 y 07).** En la parte de entrenamiento, `Replace Missing Values` (media) y `Normalize` (transformación Z) se ajustan solo con los datos de entrenamiento y se unen al modelo con `Group Models`; la parte de prueba aplica ese conjunto con `Apply Model`. El Random Forest no necesita escalado, así que en el proceso 07 solo lleva el relleno de huecos.
* **Referencias (procesos 04 y 05).** La predicción se crea con `Generate Attributes` (`pred_persistencia = readiness_hoy`, `pred_media7d = readiness_media_7d`) y se le da el rol `prediction`. En el proceso 04 van dentro de su propia `Cross Validation`, con un `Default Model` que solo está porque la `Cross Validation` exige un modelo, para que su error se calcule por jugadora igual que el de los modelos.
* **Configuración de los modelos (fija):** `Linear Regression` con `ridge = 10`; `Random Forest` con 100 árboles, profundidad máxima 10, hojas de al menos 20, división mínima 4, ganancia mínima 0,01 y semilla 1992; `Neural Net` con una capa oculta de 16 neuronas, 200 ciclos, tasa de aprendizaje 0,001 y momento 0,2.
* **Validación temporal (proceso 05).** `Filter Examples` separa `temporada = 2020` (6.231 ejemplos, entrenamiento) y `temporada = 2021` (8.400, prueba).
* **Clasificación (proceso 07).**
  * `Generate Attributes` crea `dia_malo = if(readiness_manana<=4,1,0)`, `Numerical to Binominal` lo convierte en `false`/`true` (la clase positiva es `true`) y `Set Role` lo pone como etiqueta y aparta `readiness_manana` como `metadata`.
  * El `Random Forest` usa la misma configuración que en regresión salvo el criterio (`gini_index`) y la ganancia mínima (0). La regresión logística es el operador `Logistic Regression` de H2O. Ninguno de los dos admite pesos por clase.
  * Dentro de cada `Cross Validation` hay un `Performance` que solo mide la tasa de acierto, porque Altair exige uno. Las métricas que se usan salen de juntar las predicciones de las 50 particiones (salida `test result set`) y evaluarlas con `Performance (Binominal Classification)`.
  * Las dos reglas de referencia (`readiness_hoy ≤ 4` y `readiness_media_7d ≤ 4`) se evalúan con *recall*, precisión y F1. Su AUC se calcula en Python con `src/clasificacion_referencias.py`, porque una regla fija no da una puntuación que Altair pueda ordenar.

## 3. Resultados que debe dar cada proceso

Para comprobar que la ejecución es correcta. La interpretación está en los apartados 3 a 5 del README.

### Proceso 04: validación por jugadoras

Media entre las 50 jugadoras ± desviación típica (`PerformanceVector` de cada `Cross Validation`). El r² de Altair es el cuadrado de la correlación.

| Método | RMSE | MAE | r² |
| :--- | :---: | :---: | :---: |
| Persistencia | 1,268 ± 0,438 | 0,841 ± 0,369 | 0,183 |
| Media de 7 días | 1,081 ± 0,362 | 0,783 ± 0,306 | 0,182 |
| Ridge | 1,002 ± 0,309 | 0,743 ± 0,243 | 0,253 |
| Random Forest | **0,995 ± 0,287** | **0,735 ± 0,236** | **0,265** |
| Red neuronal | 1,060 ± 0,312 | 0,797 ± 0,250 | 0,212 |

Las referencias coinciden con las de `src/contraste_jugadoras.py`, que repite esta validación en Python.

**ANOVA** sobre el RMSE de cada jugadora: F = 5,126, p = 0,001 (grados de libertad: 4 entre métodos, 245 residuales).

**T-Test** por parejas (p):

| | Persistencia | Media de 7 días | Ridge | Random Forest |
| :--- | :---: | :---: | :---: | :---: |
| Media de 7 días | 0,022 | | | |
| Ridge | 0,001 | 0,243 | | |
| Random Forest | < 0,001 | 0,193 | 0,913 | |
| Red neuronal | 0,007 | 0,762 | 0,348 | 0,280 |

### Proceso 05: validación temporal (2020 → 2021)

| Método | RMSE | MAE | r² |
| :--- | :---: | :---: | :---: |
| Persistencia | 1,194 | 0,736 | 0,477 |
| Media de 7 días | 1,027 | 0,692 | 0,552 |
| Ridge | 0,976 | 0,693 | 0,596 |
| Random Forest | **0,960** | **0,667** | **0,610** |
| Red neuronal | 1,416 | 1,059 | 0,487 |

### Proceso 06: importancia de variables y error por franja

Importancia del Random Forest de 2020 (`pesos_rf_2020`), las seis primeras: `readiness_media_propia` 0,103 · `readiness_media_7d` 0,093 · `carga_hoy` 0,085 · `readiness_hoy` 0,080 · `carga_ayer` 0,043 · `readiness_media_3d` 0,043.

Error del Random Forest en 2021 según la readiness real del día siguiente:

| Franja | Ejemplos | MAE |
| :--- | :---: | :---: |
| Baja (1-4) | 591 | 1,812 |
| Media (5-7) | 5.016 | 0,526 |
| Alta (8-10) | 2.793 | 0,679 |

### Proceso 07: clasificación de días malos

Predicciones de las 50 particiones juntas: 14.631 días, 875 días malos.

| Método | AUC | *Recall* | Precisión | F1 | Especificidad | Días malos detectados | Falsas alarmas |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Persistencia (`readiness_hoy ≤ 4`) | 0,827\* | **58,6 %** | 60,2 % | 59,4 % | 97,5 % | 513 | 339 |
| Media de 7 días ≤ 4 | 0,856\* | 45,6 % | 87,5 % | 60,0 % | 99,6 % | 399 | 57 |
| Regresión logística | 0,869 | 40,8 % | 87,1 % | 55,6 % | 99,6 % | 357 | 53 |
| Random Forest | **0,888** | 46,7 % | **89,5 %** | **61,4 %** | **99,7 %** | 409 | 48 |

\* Calculado con `src/clasificacion_referencias.py`.
