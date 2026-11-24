# Predicción de la disposición para entrenar (*readiness*) con SoccerMon — parte en Altair AI Studio

Guía de los procesos de Altair AI Studio. La organización general del proyecto (qué se hace en Altair y qué en Python) está en el `README.md`.

Proyecto práctico para la asignatura **Inteligencia Artificial en las Organizaciones (IAO, UC3M)**. El objetivo es predecir la puntuación subjetiva de disposición para entrenar (*readiness*, escala 1–10) que una futbolista declarará al día siguiente ($t+1$), a partir de su historial de bienestar y su carga de entrenamiento acumulada hasta el día actual ($t$).

---
## 1. Preparación del repositorio y ejecución en Altair AI Studio

### Requisitos previos

* **Altair AI Studio** instalado (versión 2024 o superior).
* Extensión **Python Scripting** instalada desde el *Marketplace* de Altair (`Extensions -> Marketplace -> Python Scripting`) y vinculada a un entorno de **Python 3.11+** con las librerías `pandas` y `numpy` (`Settings -> Preferences -> Python Scripting`).
### Paso 1: Vincular el proyecto como Repositorio Local
Para que todas las referencias internas de lectura y escritura (`Retrieve` y `Store`) funcionen mediante rutas relativas sin fallos entre equipos:
1. Descarga o clona este repositorio en tu equipo (por ejemplo, en `.../iao-readiness-soccermon/`).
2. Abre **Altair AI Studio** y ve al panel izquierdo **Repositories**.
3. Pulsa el icono **`+` (*Create Repository*) $\rightarrow$ *New local repository***.
4. Asigna un nombre al repositorio (por ejemplo, `Grupo10_SoccerMon`) y selecciona en *Root Directory* la carpeta raíz del proyecto (`iao-readiness-soccermon/`).
5. Comprueba que en el árbol izquierdo aparecen las carpetas `data/`, `models/`, `processes/`, `reports/` y `src/`.
### Paso 2: Configurar la ruta de ingesta (`repo_dir`) en el Proceso 01
* Si deseas ejecutar el flujo completo **desde los datos crudos** (`data/raw/subjective/`), abre el proceso `processes/01_Ingesta_y_Limpieza`, despliega el panel **Context** (*View $\rightarrow$ Show Panel $\rightarrow$ Context*) o busca en el buscador `macros` y despliega su panel para actualizar el valor de la macro **`repo_dir`** con la ruta absoluta local de tu carpeta del proyecto (**sin barra `/` ni `\` al final**).
* **Ruta de los scripts.** Los operadores `Execute Python` de los procesos 01 y 03 tienen en el parámetro *script file* una ruta absoluta del equipo donde se crearon (`/home/v1adi/.../src/carga.py` y `.../src/features.py`). En otro equipo hay que cambiarla por la ruta local de `src/carga.py` y `src/features.py`.
* *(Opcional)* El repositorio ya incluye las tablas procesadas (`data/processed/jugadora_dia` y `ejemplos`) y los modelos entrenados (`models/`), por lo que también es posible ejecutar directamente desde el **Proceso 02** o el **Proceso 04**.
### Paso 3: Orden de ejecución de los procesos
Los flujos de la carpeta `processes/` están diseñados para ejecutarse de forma secuencial (`F11` o botón *Run*), comunicándose entre sí mediante el repositorio local:

| Orden | Proceso (`.rmp`) | Qué hace | Entradas (`Retrieve` / disco) | Salidas persistidas (`Store`) |
| :---: | :--- | :--- | :--- | :--- |
| **1** | `01_Ingesta_y_Limpieza` | Integra cuestionarios subjetivos y sesiones (`src/carga.py` en `Execute Python`) en una rejilla continua por jugadora y día. | `data/raw/subjective/` (vía macro `repo_dir`) | `data/processed/jugadora_dia`<br>`data/processed/jugadoras` |
| **2** | `02_AED_Exploracion` | Calcula estadísticos descriptivos, cobertura temporal, matriz de correlaciones y autocorrelación ($t \rightarrow t+1$). | `data/processed/jugadora_dia` | Tablas y vistas exploratorias en *Results* |
| **3** | `03_Ingenieria_Caracteristicas` | Genera las 56 variables explicativas (`src/features.py` en `Execute Python`) y filtra los pares válidos hoy/mañana. | `data/processed/jugadora_dia` | `data/processed/ejemplos` |
| **4** | `04_Baselines_y_Modelado` | Evalúa *baselines* y modelos mediante CV agrupada por jugadora (`split on batch attribute`, 50 particiones) + `ANOVA` y `T-Test`. | `data/processed/ejemplos` | `models/modelo_ridge_cv`<br>`models/modelo_rf_cv`<br>`models/modelo_nn_cv` |
| **5** | `05_Validacion_Temporal_2020_2021` | Entrena exclusivamente con la temporada **2020** (6.231 ejemplos) y evalúa sobre **2021** (8.400 ejemplos). | `data/processed/ejemplos` | `models/pipeline_rf_2020`<br>`models/pesos_rf_2020` |
| **6** | `06_Explicabilidad_y_Errores` | Extrae la importancia global de variables y desglosa el error de 2021 por franja de *readiness* (baja, media, alta). | `models/pipeline_rf_2020`<br>`models/pesos_rf_2020`<br>`data/processed/ejemplos` | Tabla de pesos y métricas por zona en *Results* |

---
## 2. Conjunto de datos y planteamiento
* **Fuente:** Conjunto público **SoccerMon**, recogido durante dos temporadas (2020 y 2021) sobre dos equipos femeninos de fútbol de élite, con 33.849 registros subjetivos y 10.075 objetivos.
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
│   └── 06_Explicabilidad_y_Errores
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
### Reglas críticas y decisiones de diseño
* **Validación sin fuga entre días de una misma jugadora:** En el Proceso 4 se activa `split on batch attribute` sobre `jugadora` para que todas las observaciones de una futbolista pertenezcan exclusivamente a entrenamiento o a prueba en cada partición.
* **No imputar antes de las ventanas móviles:** Los días sin cuestionario se mantienen como ausentes hasta calcular las medias móviles de 3 y 7 días; imputar antes propagaría valores artificiales entre semanas.
* **Encapsulación del preprocesamiento (`Group Models`):** Tanto dentro de cada `Cross Validation` (Proceso 4) como en el salto temporal `2020 -> 2021` (Proceso 5), la imputación de ausentes (`Replace Missing Values`) y el escalado (`Normalize`, _Z-transformation_) se ajustan únicamente sobre entrenamiento y se aplican a prueba mediante `Group Models` + `Apply Model`.
* **Hiperparámetros fijos:** Los modelos utilizan configuraciones fijas de referencia (sin optimización automática de hiperparámetros), lo que permite comparar los algoritmos en igualdad de condiciones sin requerir validación cruzada anidada.
*(Nota: los procesos de ingesta e ingeniería de características en Altair AI Studio reutilizan directamente los scripts `src/carga.py` y `src/features.py` mediante operadores `Execute Python`).*
---
## 5. Resultados experimentales (Altair AI Studio)

### 5.1. Validación por jugadoras (Proceso 4)

Evaluación mediante validación cruzada de 50 particiones sobre los 14.631 ejemplos (criterio principal: **RMSE**):

| Método / Modelo                                                | RMSE (Media $\pm$ DE) | MAE (Media $\pm$ DE) | $r^2$ (Correlación$^2$) |
| :------------------------------------------------------------- | :-------------------: | :------------------: | :---------------------: |
| **Baseline: Persistencia** (`readiness_hoy`) ⚠️ *pendiente de recalcular en Altair* |   1,268 $\pm$ 0,438 *(Python)*   |  0,841 $\pm$ 0,369 *(Python)*  |          —          |
| **Baseline: Media de 7 días** (`readiness_media_7d`, CV batch) |   1,081 $\pm$ 0,362   |  0,783 $\pm$ 0,306   |          0,182          |
| **Regresión Ridge** (`ridge = 10.0`, fijo)                     |   1,002 $\pm$ 0,309   |    0,743 ± 0,243     |          0,253          |
| **Random Forest** (`100 trees, leaf = 20, subset = 1.0`, fijo) | **0,995 $\pm$ 0,287** |  **0,735 ± 0,236**   |        **0,265**        |
| **Red Neuronal (MLP)** (`hidden = 16, lr = 0.001`, fijo)       |   1,060 $\pm$ 0,312   |    0,797 ± 0,250     |          0,212          |

> ⚠️ **Fila de persistencia.** Antes estaba calculada con todos los días juntos (RMSE 1,237, MAE 0,765), mientras que el resto de filas es la media entre jugadoras. Así parecía mejor que la media de 7 días, cuando es al revés. Hay que calcularla en Altair dentro del mismo `Cross Validation`, igual que la media de 7 días. Mientras tanto, la fila muestra el valor calculado con la misma cuenta en Python (`src/contraste_jugadoras.py`), que reproduce los errores de Altair: media de 7 días, Ridge y Random Forest coinciden en las dos herramientas.

**Significación estadística (`ANOVA` y `T-Test` sobre RMSE entre jugadoras):**
* **ANOVA global:** $F = 0,890$, $p = 0,447$ ($DF_{\text{Between}} = 3$, $DF_{\text{Residuals}} = 196$, $DF_{\text{Total}} = 199$). No detecta diferencias estadísticamente significativas al 95 % de confianza debido a la elevada heterogeneidad natural entre jugadoras (algunas mantienen valores casi constantes mientras otras oscilan fuertemente).
* **T-Test por parejas ($\alpha = 0,05$):**
  * **Random Forest vs. Regresión Ridge:** $p = 0,913$ (empate técnico).
  * **Random Forest vs. Neural Net:** $p = 0,280$ (sin diferencia significativa).
  * **Random Forest vs. Baseline (Media 7 días):** $p = 0,193$ (sin diferencia significativa por la alta varianza inter-jugadora, aunque Random Forest reduce el RMSE medio en un 8,0 % y presenta la menor dispersión).
  * **Regresión Ridge vs. Neural Net:** $p = 0,348$ (sin diferencia significativa).
  * **Regresión Ridge vs. Baseline (Media 7 días):** $p = 0,243$ (sin diferencia significativa).
  * **Neural Net vs. Baseline (Media 7 días):** $p = 0,762$ (sin diferencia significativa).

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

---
## 6. Conclusiones principales
1. **Complementariedad entre validación por jugadoras y validación temporal:** Al evaluar sobre jugadoras completamente nuevas (`split on batch attribute`, 50 particiones), *Random Forest* (`RMSE = 0,995 ± 0,287`) y *Regresión Ridge* (`1,002 ± 0,309`) obtienen los menores errores medios, aunque la alta variabilidad individual entre futbolistas impide que las diferencias alcancen significación estadística en el contraste paramétrico ($p = 0,447$). La comparación pareada por jugadora (Wilcoxon) sí muestra que los dos mejoran a la media de 7 días ($p < 0,001$). En la validación temporal `2020 -> 2021` (Proceso 5), *Random Forest* obtiene el menor error en todas las métricas (`RMSE = 0,960`, `MAE = 0,667`, $r^2 = 0,610$) en consonancia con los resultados obtenidos en Python.
2. **Importancia del historial individual y la carga reciente:** La extracción de pesos de *Random Forest* (`AttributeWeights`) demuestra que combinar la referencia histórica propia de cada deportista (`readiness_media_propia`, `readiness_media_7d`) con la carga inmediata (`carga_hoy`, `carga_ayer`) aporta valor predictivo real frente a reglas heurísticas simples. La *Red Neuronal* no consigue aprender bien con su configuración actual (200 ciclos y tasa de aprendizaje 0,001): queda por detrás de la referencia y, en la validación temporal, de todos los métodos. Con un ajuste de sus hiperparámetros debería mejorar (en Python, con ajuste, obtiene un RMSE de 0,999 en la validación temporal).
3. **Aplicación práctica en el cuerpo técnico:** El sistema ofrece una precisión muy elevada en rangos habituales y altos de disposición (`MAE = 0,526` en zona media y `0,679` en zona alta), pero presenta mayor error en caídas abruptas (`MAE = 1,812` en zona baja). En un entorno deportivo profesional debe utilizarse como herramienta de cribado preventivo para detectar desviaciones respecto a la línea base de cada jugadora, complementada siempre con la valoración del cuerpo técnico y médico.