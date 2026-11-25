# Predicción de la disposición para entrenar (*readiness*) con SoccerMon — parte en Altair AI Studio

Guía de los procesos de Altair AI Studio. La organización general del proyecto (qué se hace en Altair y qué en Python) está en el `README.md`.

Proyecto práctico para la asignatura **Inteligencia Artificial en las Organizaciones (IAO, UC3M)**. El objetivo es predecir la puntuación subjetiva de disposición para entrenar (*readiness*, escala 1–10) que una futbolista declarará al día siguiente ($t+1$), a partir de su historial de bienestar y su carga de entrenamiento acumulada hasta el día actual ($t$).

---
## 1. Preparación del repositorio y ejecución en Altair AI Studio

### Requisitos previos

* **Altair AI Studio** instalado (versión 2024 o superior).
* Extensión **Python Scripting** instalada desde el *Marketplace* de Altair (`Extensions -> Marketplace -> Python Scripting`) y vinculada a un entorno de **Python 3.11+** con las librerías `pandas` y `numpy` (`Settings -> Preferences -> Python Scripting`).
### Paso 1: Vincular el proyecto como Repositorio Local
Los procesos leen y guardan sus datos (`Retrieve` y `Store`) dentro del repositorio `//Grupo10_SoccerMon/...`:
1. Descarga o clona este repositorio en tu equipo (por ejemplo, en `.../iao-readiness-soccermon/`).
2. Abre **Altair AI Studio** y ve al panel izquierdo **Repositories**.
3. Pulsa el icono **`+` (*Create Repository*) $\rightarrow$ *New local repository***.
4. Llama al repositorio exactamente **`Grupo10_SoccerMon`** (los `Retrieve` y `Store` de los procesos lo buscan por ese nombre) y selecciona en *Root Directory* la carpeta raíz del proyecto (`iao-readiness-soccermon/`).
5. Comprueba que en el árbol izquierdo aparecen las carpetas `data/`, `models/`, `processes/`, `reports/` y `src/`.
### Paso 2: Configurar la ruta del proyecto (`repo_dir`) en los procesos 01 y 03
* Los procesos `01_Ingesta_y_Limpieza` y `03_Ingenieria_Caracteristicas` ejecutan `src/carga.py` y `src/features.py` con `Execute Python`. Los dos toman la ruta de la macro **`repo_dir`** (el parámetro *script file* vale `%{repo_dir}/src/carga.py` y `%{repo_dir}/src/features.py`), y el 01 la usa también para leer `data/raw/subjective/`.
* En otro equipo, abre cada uno de los dos procesos, despliega el panel **Context** (*View $\rightarrow$ Show Panel $\rightarrow$ Context*) y cambia el valor de **`repo_dir`** por la ruta absoluta local de la carpeta del proyecto (**sin barra `/` ni `\` al final**).
* *(Opcional)* El repositorio ya incluye las tablas procesadas (`data/processed/jugadora_dia` y `ejemplos`) y los modelos entrenados (`models/`), por lo que también es posible ejecutar directamente desde el **Proceso 02** o el **Proceso 04**.
### Paso 3: Orden de ejecución de los procesos
Los flujos de la carpeta `processes/` están diseñados para ejecutarse de forma secuencial (`F11` o botón *Run*), comunicándose entre sí mediante el repositorio local:

| Orden | Proceso (`.rmp`) | Qué hace | Entradas (`Retrieve` / disco) | Salidas persistidas (`Store`) |
| :---: | :--- | :--- | :--- | :--- |
| **1** | `01_Ingesta_y_Limpieza` | Integra cuestionarios subjetivos y sesiones (`src/carga.py` en `Execute Python`) en una rejilla continua por jugadora y día. | `data/raw/subjective/` (vía macro `repo_dir`) | `data/processed/jugadora_dia`<br>`data/processed/jugadoras` |
| **2** | `02_AED_Exploracion` | Calcula estadísticos descriptivos, cobertura temporal, matriz de correlaciones y autocorrelación ($t \rightarrow t+1$). | `data/processed/jugadora_dia` | Tablas y vistas exploratorias en *Results* |
| **3** | `03_Ingenieria_Caracteristicas` | Genera las 56 variables explicativas (`src/features.py` en `Execute Python`) y filtra los pares válidos hoy/mañana. | `data/processed/jugadora_dia` | `data/processed/ejemplos` |
| **4** | `04_Baselines_y_Modelado` | Evalúa las dos *baselines* (persistencia y media de 7 días) y los modelos mediante CV agrupada por jugadora (`split on batch attribute`, 50 particiones) + `ANOVA` y `T-Test`. | `data/processed/ejemplos` | `models/modelo_ridge_cv`<br>`models/modelo_rf_cv`<br>`models/modelo_nn_cv` |
| **5** | `05_Validacion_Temporal_2020_2021` | Entrena exclusivamente con la temporada **2020** (6.231 ejemplos) y evalúa sobre **2021** (8.400 ejemplos). | `data/processed/ejemplos` | `models/pipeline_rf_2020`<br>`models/pesos_rf_2020` |
| **6** | `06_Explicabilidad_y_Errores` | Extrae la importancia global de variables y desglosa el error de 2021 por franja de *readiness* (baja, media, alta). | `models/pipeline_rf_2020`<br>`models/pesos_rf_2020`<br>`data/processed/ejemplos` | Tabla de pesos y métricas por zona en *Results* |
| **7** | `07_Clasificacion_Dia_Malo` | Clasifica si mañana será un día malo (*readiness* ≤ 4) con *Random Forest* y regresión logística, con la misma CV por jugadora, y lo compara con dos reglas sencillas. | `data/processed/ejemplos` | `models/modelo_rf_clasificacion_cv` |

---
## 2. Conjunto de datos y planteamiento
* **Fuente:** Conjunto público **SoccerMon**, recogido durante dos temporadas (2020 y 2021) sobre dos equipos femeninos de fútbol de élite, de los que se usa solo la parte subjetiva: 16.997 cuestionarios de bienestar con *readiness* y 16.265 sesiones de entrenamiento de 50 jugadoras.
* **Granularidad y muestra supervisada:**
  * Tabla maestra diaria: **36.550 filas** (50 futbolistas $\times$ 731 días naturales).
  * Ejemplos supervisados válidos (pares consecutivos con *readiness* conocida hoy y mañana): **14.631 ejemplos** (6.231 en la temporada 2020 y 8.400 en la temporada 2021), descritos mediante **56 variables** de contexto, historial de *readiness*, bienestar subjetivo y carga aguda/crónica (ACWR).
---
## 3. Estructura del repositorio

```text
iao-readiness-soccermon/
├── data/
│   ├── raw/subjective/          # Datos originales de SoccerMon sin modificar
│   └── processed/               # Tablas generadas (jugadora_dia, ejemplos)
├── models/                      # Modelos y pipelines persistidos en Altair
│   ├── modelo_ridge_cv
│   ├── modelo_rf_cv
│   ├── modelo_nn_cv
│   ├── pipeline_rf_2020
│   └── pesos_rf_2020
├── processes/                   # Flujos de Altair AI Studio (.rmp)
│   ├── 01_Ingesta_y_Limpieza
│   ├── 02_AED_Exploracion
│   ├── 03_Ingenieria_Caracteristicas
│   ├── 04_Baselines_y_Modelado
│   ├── 05_Validacion_Temporal_2020_2021
│   ├── 06_Explicabilidad_y_Errores
│   └── 07_Clasificacion_Dia_Malo
├── reports/                     # Gráficas vectoriales (.png) exportadas
├── src/                         # Scripts de Python (los procesos 01 y 03 ejecutan carga.py y features.py)
└── README.md                    # Documentación y guía de reproducción
```


## 4. Consideraciones metodológicas y orden de ejecución
### Orden de ejecución en Altair AI Studio
Los procesos dentro de `processes/` son secuenciales y se comunican mediante el repositorio local (`Store` / `Retrieve`):
1. **`01_Ingesta_y_Limpieza`**: Integra los cuestionarios subjetivos y la carga diaria reutilizando `src/carga.py` dentro de `Execute Python` en una rejilla continua por jugadora y fecha, guardando `data/processed/jugadora_dia`.
2. **`02_AED_Exploracion`**: Analiza distribución de la variable objetivo, patrones de valores ausentes, correlaciones y autocorrelación temporal ($t$ frente a $t+1$).
3. **`03_Ingenieria_Caracteristicas`**: Construye las **56 variables explicativas** reutilizando `src/features.py` dentro de `Execute Python` (historial de *readiness* hoy/ayer/anteayer, **medias móviles de 3 y 7 días**, media propia acumulada hasta la fecha, contexto semanal, variables de bienestar y métricas de carga diaria, aguda/crónica `ATL`, `CTL` de 28 días, `ACWR`, monotonía y *strain*) y filtra los 14.631 pares válidos en `data/processed/ejemplos`.
4. **`04_Baselines_y_Modelado`**: Evalúa los _baselines_ de referencia (_Persistencia_ y _Media de 7 días_) y ejecuta una validación cruzada agrupada por jugadora mediante `split on batch attribute` (**50 particiones Leave-One-Player-Out**, evaluando siempre sobre jugadoras no vistas en entrenamiento) sobre los tres modelos (_Regresión Ridge_, _Random Forest_ y _Red Neuronal_), realiza los contrastes estadísticos (`ANOVA` y `T-Test`) y persiste los modelos entrenados en `models/`.
5. **`05_Validacion_Temporal_2020_2021`**: Entrena los modelos exclusivamente con los datos de **2020** (6.231 ejemplos) y los evalúa sobre la temporada **2021** (8.400 ejemplos), guardando el _pipeline_ ganador (`pipeline_rf_2020`) y los pesos de importancia (`pesos_rf_2020`).
6. **`06_Explicabilidad_y_Errores`**: Reutiliza los artefactos persistidos en el Proceso 5 (sin reentrenar) para extraer la importancia global de variables mediante los pesos de *Random Forest* (`AttributeWeights`) y desglosar el error sobre 2021 por franja de _readiness_ (baja, media y alta).
7. **`07_Clasificacion_Dia_Malo`**: Convierte el problema en una clasificación binaria: `dia_malo = 1` si la *readiness* de mañana es 4 o menos (875 de 14.631 ejemplos, un 6 %). `readiness_manana` se aparta con el rol `metadata` para que ningún modelo la use como entrada. Entrena *Random Forest* (100 árboles, profundidad 10, hojas de 20 y semilla 1992, como en el Proceso 4, pero con criterio `gini_index` y `minimal_gain = 0`) y regresión logística dentro de la misma CV por jugadora (50 particiones). Se entrenan sin pesos por clase: ninguno de los dos operadores de Altair los admite. El desequilibrio (6 % de días malos) se tiene en cuenta al elegir las métricas, no la tasa de acierto. Gini es el criterio habitual para clasificar, y la ganancia mínima de 0,01 de la regresión no es comparable en clasificación: con un 6 % de días malos impediría que los árboles llegaran a separarlos. Dentro de cada CV hay un `Performance` que solo mide la tasa de acierto, porque Altair exige uno. Las predicciones de las 50 particiones se juntan (`test result set`) y se evalúan a la vez con `Performance (Binominal Classification)`: AUC, *recall*, precisión, F1, especificidad y tasa de acierto. Se juntan porque 7 jugadoras no tienen ningún día malo y en ellas el AUC y el *recall* no existen; la media por jugadora no se puede calcular. Referencias, sin entrenamiento: «mañana malo si hoy fue malo» (`readiness_hoy ≤ 4`) y «si la media de 7 días es 4 o menos», evaluadas con *recall*, precisión y F1. Una regla fija no da probabilidades y Altair no deja asignarle una puntuación, así que su AUC se calcula en Python ordenando los días por esa misma *readiness* (`src/clasificacion_referencias.py`).
### Reglas críticas y decisiones de diseño
* **Validación sin fuga entre días de una misma jugadora:** En el Proceso 4 se activa `split on batch attribute` sobre `jugadora` para que todas las observaciones de una futbolista pertenezcan exclusivamente a entrenamiento o a prueba en cada partición.
* **No imputar antes de las ventanas móviles:** Los días sin cuestionario se mantienen como ausentes hasta calcular las medias móviles de 3 y 7 días; imputar antes propagaría valores artificiales entre semanas.
* **Encapsulación del preprocesamiento (`Group Models`):** Tanto dentro de cada `Cross Validation` (Proceso 4) como en el salto temporal `2020 -> 2021` (Proceso 5), la imputación de ausentes (`Replace Missing Values`) y el escalado (`Normalize`, _Z-transformation_) se ajustan únicamente sobre entrenamiento y se aplican a prueba mediante `Group Models` + `Apply Model`.
* **Hiperparámetros fijos:** Los modelos utilizan configuraciones fijas de referencia (sin optimización automática de hiperparámetros). Se decidió así por coste: la validación por jugadora (50 particiones) ya tarda unos 10 minutos, y buscar hiperparámetros dentro de cada partición lo multiplicaría por el número de combinaciones. Es una limitación: la Red Neuronal, en particular, usa pocos ciclos (200) y una tasa de aprendizaje baja (0,001), y por eso queda por detrás de los demás modelos (ver Conclusiones).
*(Nota: los procesos de ingesta e ingeniería de características en Altair AI Studio reutilizan directamente los scripts `src/carga.py` y `src/features.py` mediante operadores `Execute Python`).*
---
## 5. Resultados experimentales (Altair AI Studio)

### 5.1. Validación por jugadoras (Proceso 4)

Evaluación mediante validación cruzada de 50 particiones sobre los 14.631 ejemplos (criterio principal: **RMSE**):

| Método / Modelo                                                | RMSE (Media $\pm$ DE) | MAE (Media $\pm$ DE) | $r^2$ (Correlación$^2$) |
| :------------------------------------------------------------- | :-------------------: | :------------------: | :---------------------: |
| **Baseline: Persistencia** (`readiness_hoy`, CV batch)        |   1,268 $\pm$ 0,438   |  0,841 $\pm$ 0,369   |          0,183          |
| **Baseline: Media de 7 días** (`readiness_media_7d`, CV batch) |   1,081 $\pm$ 0,362   |  0,783 $\pm$ 0,306   |          0,182          |
| **Regresión Ridge** (`ridge = 10.0`, fijo)                     |   1,002 $\pm$ 0,309   |    0,743 ± 0,243     |          0,253          |
| **Random Forest** (`100 trees, leaf = 20, subset = 1.0`, fijo) | **0,995 $\pm$ 0,287** |  **0,735 ± 0,236**   |        **0,265**        |
| **Red Neuronal (MLP)** (`hidden = 16, lr = 0.001`, fijo)       |   1,060 $\pm$ 0,312   |    0,797 ± 0,250     |          0,212          |

Las dos *baselines* se calculan dentro de su propio `Cross Validation` (`Cross Validation Persistencia` y `Cross Validation Baseline 7d`), como los modelos, para que todas las cifras sean la media entre las 50 jugadoras. Los valores coinciden con los de `src/contraste_jugadoras.py`, que repite esta misma validación.

**Significación estadística (`ANOVA` y `T-Test` sobre RMSE entre jugadoras):**
* **ANOVA global (5 métodos):** $F = 5,126$, $p = 0,001$ ($DF_{\text{Between}} = 4$, $DF_{\text{Residuals}} = 245$, $DF_{\text{Total}} = 249$). Hay diferencias entre los métodos, pero el *T-Test* muestra que se deben solo a la persistencia, que es peor que todos los demás. Entre los modelos y la media de 7 días no detecta diferencias, por la elevada heterogeneidad entre jugadoras (algunas mantienen valores casi constantes mientras otras oscilan fuertemente).
* **T-Test por parejas ($\alpha = 0,05$):**
  * **Random Forest vs. Regresión Ridge:** $p = 0,913$ (empate técnico).
  * **Random Forest vs. Neural Net:** $p = 0,280$ (sin diferencia significativa).
  * **Random Forest vs. Baseline (Media 7 días):** $p = 0,193$ (sin diferencia significativa por la alta varianza inter-jugadora, aunque Random Forest reduce el RMSE medio en un 8,0 % y presenta la menor dispersión).
  * **Regresión Ridge vs. Neural Net:** $p = 0,348$ (sin diferencia significativa).
  * **Regresión Ridge vs. Baseline (Media 7 días):** $p = 0,243$ (sin diferencia significativa).
  * **Neural Net vs. Baseline (Media 7 días):** $p = 0,762$ (sin diferencia significativa).
  * **Persistencia frente a los demás:** Ridge $p = 0,001$, Random Forest $p < 0,001$, Neural Net $p = 0,007$, Media 7 días $p = 0,022$ (la persistencia es significativamente peor que todos).

**Contraste pareado por jugadora (Wilcoxon, en Python: `src/contraste_jugadoras.py`).** El ANOVA y el t-test mezclan los errores de todas las jugadoras, y la diferencia entre jugadoras fáciles y difíciles tapa la diferencia entre métodos. La prueba de Wilcoxon compara los métodos jugadora a jugadora y sí detecta la mejora:
  * **Random Forest vs. Media de 7 días:** gana en 41 de 50 jugadoras, $p < 0,001$.
  * **Ridge vs. Media de 7 días:** gana en 43 de 50 jugadoras, $p < 0,001$.
  * **Random Forest vs. Ridge:** $p = 0,278$ (sin diferencia significativa).
### 5.2. Validación temporal entre temporadas (Proceso 5 — Train 2020 $\rightarrow$ Test 2021)
Entrenamiento sobre la temporada 2020 (6.231 ejemplos) y evaluación sobre la temporada 2021 (8.400 ejemplos):

| Método / Modelo               | RMSE (2021) | MAE (2021) | $r^2$ (Correlación$^2$, 2021) |
| :---------------------------- | :---------: | :--------: | :---------------------------: |
| **Baseline: Persistencia**    |    1,194    |   0,736    |             0,477             |
| **Baseline: Media de 7 días** |    1,027    |   0,692    |             0,552             |
| **Regresión Ridge**           |    0,976    |   0,693    |             0,596             |
| **Random Forest**             |  **0,960**  | **0,667**  |           **0,610**           |
| **Red Neuronal (MLP)**        |    1,416    |   1,059    |             0,487             |

### 5.3. Explicabilidad y análisis de errores (Proceso 6)
* **Variables más influyentes (Importancia de *Random Forest*, `AttributeWeights` en `pesos_rf_2020`):** El modelo apoya sus predicciones principalmente en la línea base individual de cada futbolista, su tendencia semanal y la carga externa reciente:
  1. `readiness_media_propia` (peso: **0,103**)
  2. `readiness_media_7d` (peso: **0,093**)
  3. `carga_hoy` (peso: **0,085**)
  4. `readiness_hoy` (peso: **0,080**)
  5. `carga_ayer` (peso: **0,043**) y `readiness_media_3d` (peso: **0,043**)
* **Comportamiento por zona de _readiness_ (`Aggregate` sobre 2021, $N = 8.400$):**

| Franja de *readiness* real ($t+1$) | Ejemplos ($n$) | MAE (`average(error_abs)`) | Lectura operativa |
| :--- | :---: | :---: | :--- |
| **Zona Baja (1–4)** | 591 | **1,812** | Tendencia a suavizar hacia la media en caídas bruscas o fatiga aguda. |
| **Zona Media (5–7)** | 5.016 | **0,526** | Máxima precisión en el rango habitual de entrenamiento. |
| **Zona Alta (8–10)** | 2.793 | **0,679** | Alta fiabilidad en estados óptimos de recuperación. |

### 5.4. Clasificación de días malos (Proceso 7)
Predicciones de las 50 particiones juntas (14.631 días, 875 días malos).

| Método | AUC | *Recall* | Precisión | F1 | Especificidad | Días malos detectados | Falsas alarmas |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Persistencia (`readiness_hoy ≤ 4`) | 0,827\* | **58,6 %** | 60,2 % | 59,4 % | 97,5 % | 513 | 339 |
| Media de 7 días ≤ 4 | 0,856\* | 45,6 % | 87,5 % | 60,0 % | 99,6 % | 399 | 57 |
| Regresión logística | 0,869 | 40,8 % | 87,1 % | 55,6 % | 99,6 % | 357 | 53 |
| *Random Forest* | **0,888** | 46,7 % | **89,5 %** | **61,4 %** | **99,7 %** | 409 | 48 |

\* AUC de las reglas calculado con `src/clasificacion_referencias.py` (Altair no lo calcula para una regla fija).

* El *Random Forest* es el que mejor ordena los días por riesgo (AUC 0,888) y el de mejor F1, pero por poco frente a la regla de la media de 7 días (61,4 % frente a 60,0 %).
* La persistencia es la que más días malos detecta (58,6 %), a costa de que 4 de cada 10 alarmas sean falsas.
* Ningún método detecta más de la mitad de los días malos sin disparar las falsas alarmas.

---
## 6. Conclusiones principales
1. **Complementariedad entre validación por jugadoras y validación temporal:** Al evaluar sobre jugadoras completamente nuevas (`split on batch attribute`, 50 particiones), *Random Forest* (`RMSE = 0,995 ± 0,287`) y *Regresión Ridge* (`1,002 ± 0,309`) obtienen los menores errores medios, aunque la alta variabilidad individual entre futbolistas impide que su diferencia con la media de 7 días alcance significación en el *T-Test* ($p = 0,19$ y $0,24$); el ANOVA ($p = 0,001$) solo detecta que la persistencia es peor que el resto. La comparación pareada por jugadora (Wilcoxon) sí muestra que los dos mejoran a la media de 7 días ($p < 0,001$). En la validación temporal `2020 -> 2021` (Proceso 5), *Random Forest* obtiene el menor error en todas las métricas (`RMSE = 0,960`, `MAE = 0,667`, $r^2 = 0,610$).
2. **Importancia del historial individual y la carga reciente:** La extracción de pesos de *Random Forest* (`AttributeWeights`) indica que el modelo se apoya en la referencia histórica propia de cada deportista (`readiness_media_propia`, `readiness_media_7d`) y en la carga inmediata (`carga_hoy`, `carga_ayer`); el análisis SHAP del README lo confirma y muestra en qué sentido actúa cada una. La *Red Neuronal* no consigue aprender bien con su configuración actual (200 ciclos y tasa de aprendizaje 0,001): queda por detrás de la referencia y, en la validación temporal, de todos los métodos.
3. **Aplicación práctica en el cuerpo técnico:** El sistema acierta mejor en los rangos habituales y altos de disposición (`MAE = 0,526` en zona media y `0,679` en zona alta), pero presenta mayor error en caídas abruptas (`MAE = 1,812` en zona baja). En un entorno deportivo profesional debe utilizarse como herramienta de cribado preventivo para detectar desviaciones respecto a la línea base de cada jugadora, complementada siempre con la valoración del cuerpo técnico y médico.
4. **Clasificación de días malos (Proceso 7):** como alarma de *readiness* ≤ 4, el *Random Forest* es el mejor (AUC 0,888, F1 61,4 %): el 89,5 % de sus avisos aciertan, pero detecta el 46,7 % de los días malos. Mejora por poco a la regla «media de 7 días ≤ 4» (F1 60,0 %).
