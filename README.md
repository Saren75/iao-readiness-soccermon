# Predictor de la disposición al entrenamiento en fútbol de élite

Práctica final de Inteligencia Artificial en las Organizaciones (UC3M, curso 2026/27). Grupo 10.

Este README hace de cuaderno de trabajo: sigue los apartados de la memoria y se actualiza en cada paso, para que al final redactarla sea pasar esto a limpio.

**Estado actual:** tabla base construida (fase 1). Pendiente: análisis exploratorio, baselines, modelos y explicabilidad.

## Cómo ejecutarlo

Hace falta Python 3.11 o superior.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/carga.py
```

En Windows, sustituir `.venv/bin/` por `.venv\Scripts\`.

El último comando regenera `data/processed/jugadora_dia.csv` a partir de los datos crudos.

## Estructura

```
data/raw/subjective/   datos originales de SoccerMon (parte subjetiva), sin modificar
data/processed/        tablas generadas por el código
src/                   código: carga.py (fase 1)
notebooks/             análisis exploratorio y experimentos
reports/figures/       figuras para la memoria
```

---

## 1. Introducción y objetivos

**Problema.** Predecir la disposición para entrenar (*readiness*) que una futbolista declarará mañana, a partir de su carga de entrenamiento y su bienestar de los días anteriores.

- Tipo de problema: regresión supervisada sobre datos longitudinales (una serie por jugadora).
- Variable objetivo: `readiness` del día t+1, en escala de 1 a 10. Es una respuesta real del cuestionario diario, no una fórmula calculada a partir de las entradas.
- Entradas: bienestar y carga de los días hasta t.

**Objetivos.**

1. Construir y comparar modelos supervisados que mejoren a referencias sencillas (baselines).
2. Validar sin fuga de información: por jugadoras y en el tiempo.
3. Explicar qué factores pesan más en la predicción (SHAP).
4. Extensión opcional: recomendación de carga a partir de la readiness predicha.

## 2. Planteamiento y desarrollo

### 2.1 Datos

**SoccerMon** (Midoglu et al., 2024): dos equipos de la liga femenina noruega de élite, temporadas 2020 y 2021. Datos reales, recogidos con la aplicación PmSys. Solo se usa la parte subjetiva (cuestionarios); los datos de GPS quedan fuera.

| Aspecto | Valor comprobado |
|---|---|
| Jugadoras | 50 (27 del equipo A, 23 del equipo B) |
| Periodo | 1-1-2020 a 31-12-2021 (731 días) |
| Cuestionarios de bienestar | 16.997 con readiness (46,5 % de los días-jugadora) |
| Días con readiness por jugadora | de 23 a 646 |
| Sesiones de entrenamiento | 16.265 |
| Pares (día t, día t+1) con readiness en ambos | 14.631 |

#### Qué se usa y qué no

| Parte del dataset | Contenido | ¿Se usa? |
|---|---|---|
| `wellness/` (7 ficheros) | Cuestionario diario: readiness, fatiga, ánimo, horas y calidad de sueño, dolor muscular, estrés | Sí. Readiness es el objetivo; el resto, entradas |
| `training-load/session.json` | Cada sesión: duración, esfuerzo percibido (RPE) y su producto (sRPE) | Sí, resumido por día |
| `training-load/` (8 CSV) | Carga diaria y siete métricas calculadas a partir de ella por los autores | Sí, como entradas |
| `injury/`, `illness/`, `game-performance/` | 161 avisos de lesión, 14 de enfermedad y 247 valoraciones de partido | No por ahora: son informes puntuales y el alcance acordado es bienestar y carga |
| Datos objetivos (GPS, 99 GB) | Posición y movimiento en entrenamientos y partidos | No: fuera del alcance acordado |

#### Escalas y su sentido

- Readiness: de 1 («nada preparada») a 10 («con muchas ganas»).
- Fatiga, ánimo, calidad del sueño, dolor muscular y estrés: de 1 a 5.
- Duración del sueño: horas.

**En todas las escalas un valor alto es mejor.** Es fácil equivocarse con los nombres: una fatiga de 5 significa «muy fresca», no «muy cansada», y lo mismo ocurre con el dolor muscular y el estrés.

Cómo se ha comprobado:

- La Figura 2 del artículo muestra las pantallas del cuestionario: fatiga va de 1 «muy cansada» a 5 «muy fresca», y calidad del sueño de 1 «insomnio» a 5 «sueño excelente». Las pantallas de dolor muscular, estrés y ánimo no aparecen en la figura.
- Con los datos: las seis variables correlacionan en positivo con la readiness del mismo día (Spearman), también al comparar a cada jugadora consigo misma.

| Variable | Correlación con readiness | Dentro de cada jugadora |
|---|---|---|
| Fatiga | +0,48 | +0,42 |
| Dolor muscular | +0,36 | +0,31 |
| Ánimo | +0,23 | +0,21 |
| Calidad del sueño | +0,20 | +0,16 |
| Estrés | +0,19 | +0,17 |
| Horas de sueño | +0,18 | +0,10 |

Además, cuanto mayor fue la carga del día anterior, más bajos salen hoy el dolor muscular (−0,18) y la fatiga (−0,11), lo que encaja con que un valor bajo signifique «peor».

### 2.2 Fase 1: tabla jugadora-día

`src/carga.py` une los 15 ficheros CSV y el JSON de sesiones en una tabla con **una fila por jugadora y día** (36.550 filas, 23 columnas): `data/processed/jugadora_dia.csv`.

| Grupo | Columnas |
|---|---|
| Identificación | `jugadora`, `equipo`, `temporada`, `fecha` |
| Bienestar | `readiness`, `fatiga`, `animo`, `sueno_horas`, `sueno_calidad`, `dolor_muscular`, `estres`, `tiene_bienestar` |
| Sesiones del día | `n_sesiones`, `duracion_min`, `rpe_medio` |
| Carga (ya calculada en el dataset) | `carga_diaria`, `carga_semanal`, `atl`, `ctl28`, `ctl42`, `acwr`, `monotonia`, `strain` |

Decisiones y su justificación:

- **Calendario completo.** Se guardan todos los días de cada jugadora aunque no haya datos. Así la fila anterior es siempre «ayer» y los desfases no se equivocan con los huecos.
- **Día sin cuestionario = vacío.** No se rellena ni se inventa ningún valor.
- **Solo se reorganiza lo medido.** En esta fase no se crea el objetivo ni ninguna característica nueva.
- **Las métricas de carga usan solo el pasado.** Comprobado: la carga semanal es la suma de los últimos 7 días, ATL la media de 7, CTL la de 28 o 42 y ACWR es ATL entre CTL28. Todas llegan hasta el día t, así que son entradas válidas.

### 2.3 Fases siguientes

Pendiente: análisis exploratorio, baselines, objetivo y características (desfases de 1 y 2 días, ventanas de 3 y 7 días), modelos y validación.

## 3. Resultados

Referencias preliminares, calculadas sobre los 14.631 pares de días consecutivos (pendiente de repetirlas dentro del esquema de validación):

| Baseline | MAE | RMSE |
|---|---|---|
| Persistencia (mañana = hoy) | 0,765 | 1,237 |
| Media de la jugadora (orientativo: usa todos sus datos) | 0,852 | 1,171 |
| Media global | 1,130 | 1,463 |

La readiness no cambia de un día al siguiente en el 47,8 % de los casos.

## 4. Análisis y discusión

Limitaciones conocidas hasta ahora:

- **Huecos.** Más de la mitad de los días-jugadora no tiene cuestionario, y el reparto entre jugadoras es muy desigual.
- **Ceros ambiguos en la carga.** Un 0 no distingue «no entrenó» de «no lo registró».
- **Pequeñas incoherencias en el origen.** En 110 de 14.197 días con sesión, la carga diaria del dataset no coincide con la suma de sus sesiones; hay 76 sesiones repetidas; el strain no se reproduce exactamente con la fórmula habitual.
- **Datos subjetivos** de solo dos equipos femeninos de una liga.

## 5. Conclusiones

Pendiente.

## 6. Referencias

- Midoglu, C. et al. (2024). A large-scale multivariate soccer athlete health, performance, and position monitoring dataset. *Scientific Data*, 11, 553. https://doi.org/10.1038/s41597-024-03386-x
- Midoglu, C. et al. (2022). SoccerMon [conjunto de datos]. Zenodo. https://doi.org/10.5281/zenodo.10033832
- Impellizzeri, F. M. et al. (2004). Use of RPE-based training load in soccer. *Medicine & Science in Sports & Exercise*, 36, 1042–1047.

## Licencia de los datos

Los ficheros de `data/raw/subjective/` pertenecen al conjunto de datos SoccerMon, de Midoglu et al., publicado con licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Se incluyen sin modificar.
