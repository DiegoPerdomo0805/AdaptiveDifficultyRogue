# Adaptive Difficulty Rogue — DDA guiada por resultados frente a heurística

Prototipo de videojuego *roguelike* de vista cenital (pygame) que se usa como banco de pruebas de
**ajuste dinámico de dificultad** (DDA, *dynamic difficulty adjustment*). El proyecto compara tres
formas de decidir la dificultad de cada sala:

- una dificultad fija (`fixed_default`);
- una heurística basada en reglas (`heuristic`);
- una **cGAN guiada por resultados** que aprende de lo que realmente ocurre en el juego.

La comparación se hace bajo semillas y mapas equivalentes.

> **Estado actual.** El experimento `exp_v3.2` está **CONGELADO** y la **recolección principal
> todavía no ha comenzado**. Por eso este documento distingue tres niveles:
> - la **verificación funcional**: pruebas automáticas y comprobaciones de integridad;
> - los resultados de **piloto/smoke**: escala pequeña, solo para validar el pipeline;
> - los **resultados finales**: **aún no existen**.
>
> Ningún resultado *smoke* debe citarse como evidencia final.

## Contenido

1. [Problema de investigación](#1-problema-de-investigación)
2. [Objetivo del prototipo](#2-objetivo-del-prototipo)
3. [Alcance implementado](#3-alcance-implementado)
4. [Elementos descartados o acotados](#4-elementos-descartados-o-acotados)
5. [Arquitectura general](#5-arquitectura-general)
6. [Flujo de datos y telemetría](#6-flujo-de-datos-y-telemetría)
7. [Controladores](#7-controladores)
8. [Papel de Q(x,a)](#8-papel-de-qxa)
9. [Diseño de la cGAN](#9-diseño-de-la-cgan)
10. [Por qué la cGAN no imita a la heurística](#10-por-qué-la-cgan-no-imita-a-la-heurística)
11. [Métrica principal B_r](#11-métrica-principal-b_r)
12. [Selección de checkpoints en lazo cerrado](#12-selección-de-checkpoints-en-lazo-cerrado)
13. [Sistema de botín](#13-sistema-de-botín)
14. [Instalación (WSL/Linux)](#14-instalación-wsllinux)
15. [Dependencias](#15-dependencias)
16. [Cómo ejecutar el juego](#16-cómo-ejecutar-el-juego)
17. [Cómo ejecutar las pruebas](#17-cómo-ejecutar-las-pruebas)
18. [Verificación de la congelación](#18-verificación-de-la-congelación)
19. [Comandos del experimento principal](#19-comandos-del-experimento-principal)
20. [Particiones de semillas y prevención de fugas](#20-particiones-de-semillas-y-prevención-de-fugas)
21. [Resultados smoke verificados de exp_v3.2](#21-resultados-smoke-verificados-de-exp_v32)
22. [Limitaciones](#22-limitaciones)
23. [Estructura del repositorio](#23-estructura-del-repositorio)
24. [Reproducibilidad](#24-reproducibilidad)
25. [Uso de herramientas de IA en el desarrollo](#25-uso-de-herramientas-de-ia-en-el-desarrollo)
26. [Autoría, propósito académico y licencia](#26-autoría-propósito-académico-y-licencia)

---

## 1. Problema de investigación

Las heurísticas de DDA ajustan la dificultad con reglas fijas escritas a mano: más enemigos cuando
el jugador mata mucho, menos cuando tiene poca vida. Estas reglas no comprueban si el ajuste
produjo realmente la experiencia deseada. Un modelo entrenado para **imitar** esas reglas solo
reproduciría su comportamiento.

La pregunta de investigación es si un modelo generativo condicional, entrenado con los
**resultados observados** de cada intervención, puede elegir intervenciones que mantengan las
salas dentro de rangos de equilibrio declarados, y hacerlo mejor que una dificultad fija y que una
heurística.

## 2. Objetivo del prototipo

- Ofrecer un juego jugable y un bot sin interfaz que comparten exactamente la misma telemetría.
- Aprender `Q(x,a) → resultado` a partir de datos observados.
- Entrenar un generador `G(x,z) → acción` guiado por `Q`, con restricciones de plausibilidad y de
  soporte de datos.
- Evaluar los controladores con una métrica por corrida predeclarada (`B_r`), semillas pareadas e
  intervalos de confianza.

## 3. Alcance implementado

- Adaptación, guiada por resultados, de cuatro parámetros enemigos: vida (`enemy_hp_mult`), daño
  (`enemy_dmg_mult`), velocidad (`enemy_speed_mult`) y número de enemigos (`spawn_count`).
- Adaptación de la **categoría** de botín (arma, hechizo, armadura, botas), **solo si** su
  compuerta predictiva pasa con los datos principales. Actualmente está deshabilitada.
- Definiciones y estadísticas de ítems **fijas**. Los efectos ya declarados, como la distancia de
  *dash* de las botas, se volvieron funcionales en el juego y en el bot.
- Evaluación diferida del botín sobre las **3 salas de combate siguientes**.
- Una **política de jefe compartida y determinista** para todas las condiciones.

## 4. Elementos descartados o acotados

- **Descartados:**
  - adaptar el tamaño o la *hitbox* de los enemigos;
  - generar o aprender estadísticas de ítems;
  - nuevas categorías de ítems o nuevos sistemas de juego.
- **Acotados:**
  - el combate del bot es un **resolvedor abstracto**, sin física ni posiciones;
  - el jefe no lo decide ningún controlador experimental;
  - `learned_loot` está deshabilitado mientras su compuerta no pase con los datos principales.

## 5. Arquitectura general

| Capa | Archivos | Función |
|---|---|---|
| Juego | `rogue.py`, `node_map_gen.py` | juego pygame, mapa de salas y DFS |
| Núcleo compartido | `dda_core.py` | constantes, vector de condición, ítems y sus efectos, heurística, límites de acción, modificadores del jefe, `config_hash` |
| Simulación | `bot_runner.py`, `controllers.py`, `rewards.py` | bot sin interfaz, controladores enemigos, políticas de botín |
| Telemetría | `telemetry.py`, `loot_dataset.py` | esquema `room_samples_v4.0`, manifiestos, validación, dataset diferido de botín |
| Configuración | `experiment_config.json`, `balance.py`, `freeze_experiment.py`, `experiment_freeze.json` | parámetros versionados, función de equilibrio, auditoría, congelación |
| Aprendizaje | `ml_data.py`, `models.py`, `train_q.py`, `train_cgan.py`, `selection.py`, `train_q_loot.py`, `train_loot_policy.py`, `model_runtime.py` | datos, redes, entrenamiento, selección en lazo cerrado, inferencia segura |
| Evaluación | `evaluate_controllers.py`, `analyze_results.py`, `analyze_loot.py`, `dataset_tools.py` | comparación pareada, métricas, diagnósticos |
| Orquestación | `scripts/01_smoke.sh` … `06_analyze.sh`, `scripts/common.sh` | etapas reanudables con registros y marcadores `DONE` |

## 6. Flujo de datos y telemetría

```text
mapa + semilla → sala → vector de condición x (9 dim, previo a la sala)
     → controlador (o política de jefe compartida) → acción cruda a
     → finalize_action (límites, spawn entero) → effective_applied_config (modificadores de jefe)
     → combate (juego o bot) → resultado de la sala → fila RoomSample
```

**Vector de condición** (orden fijo, `COND_DIM = 9`, contrato v1.2):

```text
[melee_ratio, magic_ratio, hits_per_kill_melee, hits_per_kill_magic, damage_taken/dealt,
 hp_ratio, total_kills_norm (acumulado), node_depth_norm, progress_norm]
```

- Las componentes de comportamiento vienen de una ventana de las últimas 3 salas.
- La vida es instantánea.
- Muertes, profundidad y progreso son contextuales.
- Nunca entra información posterior a la sala.

**Acción** (límites de diseño):

| Parámetro | Rango |
|---|---|
| `enemy_hp_mult` | [0,55; 3,0] |
| `enemy_dmg_mult` | [0,45; 2,4] |
| `enemy_speed_mult` | [0,80; 1,70] |
| `spawn_count` | {2…10}, entero |

**Telemetría** (`telemetry.py`, esquema `room_samples_v4.0`). Cada sala produce una fila con:
- la identidad de la corrida: `run_id` y `run_seed`;
- las condiciones experimentales: `enemy_controller` y `reward_policy`;
- `player_profile`, solo como metadato (nunca es entrada de un modelo);
- el estado previo a la sala y el equipo;
- el vector de condición;
- la decisión: quién decidió, la versión, el checkpoint, el *fallback* y su motivo, y la distancia
  de soporte;
- la acción cruda y la efectiva;
- el evento de botín, si hubo;
- el resultado de la sala;
- las versiones.

El juego y el bot construyen las filas con las **mismas** funciones. `validate_dataset()` rechaza:
- esquemas mezclados;
- *hashes* de configuración distintos;
- claves `(run_id, room_seq)` duplicadas;
- filas incoherentes, como un jefe no decidido por `shared_boss`.

## 7. Controladores

| Controlador | Comportamiento |
|---|---|
| `fixed_default` (`fixed`) | Acción constante: multiplicadores 1,0 y `spawn_count` = 2 (`DDA_SPAWN_BASE`). Es una configuración estática canónica, **no ajustada** con resultados piloto y **muy fácil a propósito**. |
| `heuristic` | `rule_based_dda`: escala con las muertes acumuladas y la profundidad, y relaja la dificultad con poca vida. Sirve de línea base, de ancla de exploración y de *fallback* en tiempo de ejecución. **Nunca es una etiqueta de entrenamiento.** |
| `exploration` | Solo para recolectar datos. Mezcla con semilla: 70 % perturbaciones log-normales alrededor de la heurística, 20 % muestras uniformes en el rango válido y 10 % acciones de referencia. |
| `cgan` | Generador guiado por resultados (secciones 8–12). Ante una incompatibilidad o una acción fuera de soporte usa la heurística y **registra el motivo**. |

En las **salas de jefe** siempre decide `shared_boss`: la acción cruda 1,5 / 1,3 / 1,15 / 3 recibe
los modificadores de jefe (vida ×4, daño ×1,7, velocidad ×0,9, un jefe). En las salas sin enemigos
no se invoca ningún modelo.

## 8. Papel de Q(x,a)

`Q(x,a)` predice el resultado **observado** de una sala ordinaria:
- la probabilidad de muerte;
- la vida tras la sala;
- `damage_share`, el daño recibido / (recibido + infligido);
- `combat_efficiency`, las bajas / max(golpes, 1);
- `duration_norm`, min(t / 30 s, 1).

Detalles del modelo:
- Se entrena **solo** con tuplas `(x, a, resultado)` observadas, con particiones por `run_id`
  completo y escaladores ajustados únicamente con los datos de entrenamiento.
- Todas sus salidas están **acotadas por arquitectura** a [0, 1].
- Se compara con la ablación `Q(x)` (sin la acción) y con predictores ingenuos.
- **Compuerta:** `Q(x,a)` debe superar a la media y a la prevalencia en las salidas clave, y su
  ganancia sobre `Q(x)` debe ser de al menos 0,02. Si no la supera, la cGAN no se entrena.
- Durante el entrenamiento del generador, `Q` está **congelada**.

## 9. Diseño de la cGAN

```text
G(x, z) → a     generador; salida confinada a la caja de soporte observada (cuantiles 1 %–99 %)
D(x, a) → logit discriminador: pares observados frente a generados
Q(x, a) → o     evaluador congelado

L_G = L_adv + λ_bal·L_equilibrio(Q(x, G(x,z))) + λ_div·L_diversidad + λ_sup·L_soporte
```

- λ_bal = 10, λ_div = 1 (τ = 0,05) y λ_sup = 5.
- `L_soporte` penaliza los pares `(x, a)` cuya distancia kNN (k = 10) a los pares de entrenamiento
  supera el percentil 95 de las distancias de los propios datos.
- En tiempo de ejecución, `model_runtime.py` comprueba esquemas, columnas, dimensiones, límites,
  *hashes* y escaladores. Rechaza acciones con razón de soporte > 1,5 y entonces usa la heurística
  y registra `ood_action`.

## 10. Por qué la cGAN no imita a la heurística

- **No hay objetivo de reconstrucción** contra las acciones heurísticas (ni MSE ni MAE). La función
  `generator_loss` no recibe acciones observadas.
- **Las acciones heurísticas no son etiquetas.** Solo aparecen, mezcladas con las de exploración y
  las fijas, como muestras de la región plausible que ve el discriminador.
- **El generador se guía por resultados observados**, a través de la `Q` congelada y de los
  intervalos de equilibrio declarados.
- **Las restricciones adversarial y de soporte** impiden explotar extrapolaciones poco fiables de
  `Q`. La selección final, además, se basa en partidas reales (sección 12), no en predicciones.

Una prueba automática verifica que el objetivo del generador no contiene ningún término de
reconstrucción.

## 11. Métrica principal B_r

Pérdida de equilibrio (pseudo-Huber con escalas por lado acotadas):

```text
término_i = w_i · 2(√(1 + (d/s)²) − 1),   L = Σ término_i / Σ w_i
```

- `d` es la distancia fuera del intervalo (0 dentro).
- `s` es la escala configurada, acotada en cada lado por la distancia entre el límite del
  intervalo y el límite físico.
- La pérdida es 0 dentro del intervalo, cuadrática cerca de él y lineal lejos.

| Resultado | Intervalo objetivo (intención de diseño) | Peso |
|---|---|---|
| muerte | [0,00; 0,08] | 3 |
| vida tras la sala | [0,35; 0,85] | 2 |
| `damage_share` | [0,09; 0,31] | 1 |
| eficiencia | [0,15; 0,50] | 0,5 |
| duración | [0,10; 0,50] | 0,5 |

**B_r** es la métrica principal, calculada **por corrida**. Se agregan solo las salas de combate
ordinarias (sin jefe ni salas vacías):
- la tasa de muerte;
- la vida media tras cada sala (0 si la sala fue mortal);
- `damage_share` a partir de los **totales** de daño;
- la eficiencia a partir de los **totales** de bajas y golpes;
- la duración media.

`B_r` es la pérdida de ese vector. **Cuanto menor, más cerca del equilibrio deseado.**

Cada corrida es una unidad experimental, y se informan la media, la mediana, la desviación típica,
el IQR, intervalos *bootstrap* al 95 % y diferencias pareadas por semilla. La pérdida por sala solo
se usa como diagnóstico. Los resultados del jefe y la victoria se informan aparte, como
consecuencias de extremo a extremo.

## 12. Selección de checkpoints en lazo cerrado

Los diagnósticos por sala (validez, diversidad, sensibilidad y época mínima) solo **filtran**
candidatos. Cada 5 épocas se guarda un candidato, y pasan los 6 mejores. Cada candidato se recarga
por `model_runtime` y el bot **lo juega** en 200 semillas fijas de `selection_validation`, con la
política de botín heurística y el jefe compartido.

```text
selection_score = media(B_r) + 0,5 · CVaR_0,90(B_r)        (menor es mejor)
CVaR_0,90 = media de los ⌈0,1·n⌉ peores B_r
```

**Compuertas de elegibilidad** (todas obligatorias):

- validez, diversidad y sensibilidad a nivel de sala;
- acciones válidas y recarga correcta por `model_runtime`;
- tasa de muerte observada en salas ordinarias ≤ 0,08;
- todo *fallback* registrado y de tipo OOD;
- **exceso OOD relativo a la acción** ≤ 0,05;
- tasa del guardia de ejecución (> 1,5× el umbral de soporte) ≤ 0,01;
- tasa de *fallback* ≤ 0,01.

El exceso OOD es la tasa OOD de la cGAN **menos** la de la acción heurística **contrafactual en los
mismos estados**. Ese contrafactual se calcula después, a partir de las filas registradas, y no
altera la partida. La tasa OOD absoluta es solo diagnóstica.

Se elige el candidato elegible con menor puntuación. Si no hay ninguno elegible, la selección
**falla explícitamente** y no se produce checkpoint. Las predicciones de `Q` por corrida (sesgo,
MAE y correlación) se informan solo como diagnóstico.

## 13. Sistema de botín

**Efectos compartidos.** El juego (`rogue.Player`) y el bot usan las mismas funciones de
`dda_core`:

| Categoría | Ítems (estadísticas fijas) | Efecto |
|---|---|---|
| arma | Espada 14/1,00/1,00; Hacha 20/1,12/1,05; Estoque 10/0,80/0,95 (daño / enfriamiento / curación) | daño cuerpo a cuerpo, enfriamiento 0,45·cd, curación int(8·mult) por baja |
| hechizo | Bolt 10/1,00/12; Spike 8/0,78/16; Hex Bomb 30/1,25/9 (daño / enfriamiento / munición) | daño del proyectil, enfriamiento 0,60·cd, munición por sala |
| armadura | Cloth 0,5/+0; Chain 2,0/+0; Blood Harness 1,0/+6 (absorción / curación extra) | absorción fija por golpe y curación extra por baja |
| botas | Leather 1,00; Sprint 1,25; Voidstep 1,45 (`dash_dist_mult`) | distancia de *dash* de 72 / 90 / 104 px |

En el bot, un *dash* evade la embestida enemiga con probabilidad
min(1, distancia de *dash* / alcance de la embestida). Es una aproximación documentada; no existe
ninguna estadística exclusiva del bot.

**Registro y resultados diferidos:**

- Cada sala de botín registra:
  - la política que propuso las probabilidades de categoría;
  - las 3 ofertas, con identificadores estables;
  - el mecanismo de selección, que se registra aparte de la política;
  - el ítem elegido;
  - el equipo antes y después.
- `loot_dataset.py` deriva, por evento, el resultado de las **3 salas de combate siguientes** de la
  misma corrida. Se detiene antes si el jugador muere, si gana o si termina el registro.
- `analyze_loot.py` informa asociaciones **descriptivas**, no efectos causales.

**`learned_loot` está deshabilitado.** Se entrenará solo si, con los datos principales, la
compuerta predefinida de `train_q_loot.py` pasa. Esa compuerta compara cada modelo con información
de recompensa contra su gemelo sin ella:
- categoría frente a solo estado;
- categoría + ítem + equipo previo frente a estado + equipo previo.

Exige una ganancia media ≥ 0,01 y una ganancia positiva en **cada** una de las particiones 0, 1 y
2. La recolección principal rechaza `learned_loot`.

## 14. Instalación (WSL/Linux)

```bash
git clone https://github.com/DiegoPerdomo0805/AdaptiveDifficultyRogue.git
cd AdaptiveDifficultyRogue
git switch cGAN_vs_heuristic
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Solo CPU (descarga más pequeña): instalar torch primero desde el índice CPU
#   python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```

Probado con Python 3.12.3 en Linux/WSL2.

## 15. Dependencias

| Paquete | Uso |
|---|---|
| `pygame` 2.6.1 | juego |
| `torch` 2.14.1 | entrenamiento e inferencia |
| `numpy` 2.5.3 | interoperabilidad de torch |
| `matplotlib` 3.11.2 | gráficos (opcional) |
| `pytest` 9.1.1 | pruebas |

La ruta de generación de datos (bot, telemetría, controladores, políticas, equilibrio y dataset de
botín) solo usa la biblioteca estándar.

## 16. Cómo ejecutar el juego

```bash
python rogue.py
ROGUE_CONTROLLER=heuristic python rogue.py
ROGUE_CONTROLLER=cgan ROGUE_MODEL_DIR=<dir_modelo> python rogue.py
ROGUE_REWARD_POLICY=fixed_uniform python rogue.py
```

- `ROGUE_CONTROLLER` acepta `fixed`, `heuristic` o `cgan`; por defecto, `heuristic`.
- `ROGUE_REWARD_POLICY` acepta `fixed_uniform`, `heuristic` o `learned_loot`; por defecto,
  `heuristic`.

**Controles:**

| Acción | Tecla |
|---|---|
| moverse | WASD |
| cuerpo a cuerpo | clic izquierdo |
| magia | clic derecho |
| *dash* | ESPACIO |
| defender | SHIFT |
| interactuar | E |
| elegir botín | flechas y Enter |

Las sesiones humanas se guardan en `human_sessions/` con el mismo contrato que el bot, y se
analizan **por separado**.

## 17. Cómo ejecutar las pruebas

```bash
pytest -q              # 128 pruebas
python -m compileall -q .
```

## 18. Verificación de la congelación

```bash
python freeze_experiment.py verify
# freeze OK: exp_v3.2 at 22d398b14246 (config 1a6a751469dcedd1, bot cec025d6592e53a4)
```

`experiment_freeze.json` fija:
- el commit congelado;
- las versiones;
- los *hashes*;
- los objetivos, los pesos, la pérdida y la métrica principal;
- la regla de selección y sus compuertas;
- `fixed_default` y el jefe compartido;
- las mezclas de recolección;
- las compuertas de `Q` y `Q_loot`, con sus particiones;
- las semillas, los límites de soporte y el horizonte de 3 salas.

`scripts/02_collect.sh` se niega a recolectar si algo cambió. La congelación anterior se conserva en
`freeze_history/experiment_freeze_exp_v3.1.json`.

## 19. Comandos del experimento principal

**Todavía no se han ejecutado.**

```bash
python freeze_experiment.py verify && pytest -q
EXP=experiments/$(date +%Y%m%d)_main
scripts/02_collect.sh $EXP                     # verifica la congelación; mezclas de la configuración
Q_SEEDS="0 1 2" DEVICE=cuda scripts/03_train_q.sh $EXP
SEEDS="0 1 2" BATCHES="64 128" DEVICE=cuda scripts/04_train_cgan.sh $EXP
EVAL_RUNS=400 scripts/05_evaluate.sh $EXP      # solo semillas final_test
scripts/06_analyze.sh $EXP
```

- **Mezclas de recolección:** enemigos `exploration=0.65, heuristic=0.20, fixed=0.15`; botín
  `exploration=0.60, heuristic=0.20, fixed_uniform=0.20`.
- **`04_train_cgan.sh`** selecciona en lazo cerrado y entrena `learned_loot` solo si su compuerta
  pasa.
- **Funcionamiento de las etapas:** cada etapa escribe `meta.json` (commit y *hashes*), `log.txt` y
  un marcador `DONE` cuando termina. Las etapas completas se omiten al reanudar, y las incompletas
  se rechazan hasta revisarlas.
- **Ejecuciones largas:** conviene usar `tmux`.

## 20. Particiones de semillas y prevención de fugas

| Bloque | Semillas | Uso |
|---|---|---|
| `pilot` | 100 000–199 999 | pilotos |
| `train_collection` | 1 000 000–4 999 999 | datos de entrenamiento de Q, G, D y Q_loot |
| `selection_validation` | 8 000 000–8 999 999 | selección de checkpoints (las primeras 200) |
| `used_in_pilot_smoke_evaluations` | 9 000 000–9 099 999 | consumidas en evaluaciones antiguas; retiradas |
| `smoke_test` | 9 100 000–9 199 999 | comparaciones *smoke* previas a la congelación |
| `final_test` | 9 500 000–9 999 999 | **solo** la comparación final |

**Garantías:**
- La configuración valida que los bloques no se solapen.
- `selection.py` rechaza las semillas de prueba y de entrenamiento.
- `evaluate_controllers.py` solo acepta `final_test` o `smoke_test`, y rechaza solapamientos con
  los manifiestos de entrenamiento.
- Las semillas `final_test` no se usan para entrenar, ajustar, elegir épocas o checkpoints, ni
  cambiar objetivos.

**Otras protecciones:**
- Las particiones de entrenamiento se hacen por `run_id` completo.
- Ningún vector de condición contiene información posterior a la sala.
- La corrida `i` usa `run_seed = seed + i`, con flujos aleatorios derivados e independientes.

## 21. Resultados smoke verificados de exp_v3.2

> **Son resultados de validación del pipeline a escala piloto, no evidencia final.** Usan un piloto
> de 900 corridas y 200 corridas de evaluación en semillas `smoke_test` nunca usadas antes. Los
> resultados finales provendrán de la recolección principal y de las semillas `final_test`, y
> **aún no existen**.

**Verificación funcional:**
- 128 pruebas aprobadas.
- Recolección idéntica byte a byte con 1 y con 4 *workers*.
- 9 datasets *smoke* validados.
- Auditoría analítica de la función de equilibrio aprobada.
- Compuerta de `Q` aprobada.
- 18 candidatos cGAN evaluados en lazo cerrado; 7 elegibles.

**Checkpoint seleccionado:** s0 / época 25, ID `f238533b2eae852f`.

**Comparación pareada en semillas `smoke_test`** (9 100 000–9 100 199):

| Métrica | cGAN | heurística | `fixed_default` |
|---|---|---|---|
| media de B_r | **0,061** | 0,176 | 0,261 |
| CVaR₀,₉₀ de B_r | 0,229 | 0,385 | 0,365 |
| corridas con B_r > 1 | 0 % | 0 % | 0 % |
| tasa de muerte en salas ordinarias / en profundidad 1 | 0 / 0 | 4,9 % / 0 | 0 / 0 |
| victoria | 0,86 | 0,705 | 0,985 |

**Diferencias pareadas de B_r para la cGAN** (bootstrap 95 %):
- frente a la heurística: −0,115 [−0,132; −0,098];
- frente a la fija: −0,199 [−0,210; −0,189].

**Victoria:**
- La cGAN gana más que la heurística: +0,155 [0,085; 0,220].
- La cGAN gana menos que la fija: −0,125 [−0,175; −0,075].
- En la fija, el jugador llega al jefe con vida casi completa.

**Soporte:** exceso OOD relativo a la acción de +0,022; ninguna violación del guardia de ejecución;
ningún *fallback*.

**Cola catastrófica:** la configuración `exp_v3.1`, con selección por sala, tenía un 13 % de
corridas con B_r > 1 y un 7,4 % de muertes en salas de profundidad 1. Con la selección en lazo
cerrado, ninguna corrida *smoke* supera B_r = 1.

**Advertencia diagnóstica conservada.** `fixed_default` es estático y muy fácil a propósito: gana el
98,5 % de las corridas. Por eso `dataset_tools.py` marca su dataset de evaluación con
"run outcomes collapsed".
- Se mantiene como **advertencia diagnóstica no bloqueante**; no se oculta ni se desactiva.
- Afecta solo a datasets compuestos únicamente por `fixed_default`. La recolección principal es
  mixta: su piloto tuvo un 78,2 % de victorias y ninguna alerta.

## 22. Limitaciones

- El combate del bot es abstracto: no hay posiciones, combate un enemigo a la vez y el *dash* se
  aproxima como evasión. Los resultados describen el entorno del bot. Las sesiones humanas son
  otro conjunto, que todavía no se ha recogido con el esquema actual.
- `Q` subestima el B_r observado de la cGAN (*smoke*: 0,009 predicho frente a 0,061 observado).
  Por eso la selección usa partidas reales.
- La cGAN reduce la vida con la que se llega al jefe (0,79 frente a 0,84 de la heurística), porque
  sus objetivos de sala buscan vida intermedia.
- `node_depth_norm` se satura en profundidad 10, y `total_kills_norm` en 30 bajas.
- La evidencia sobre `learned_loot` fue inconclusa en los pilotos.

## 23. Estructura del repositorio

```text
*.py (raíz)             pipeline activo (rutas fijas hasta terminar el experimento)
experiment_config.json  configuración congelada exp_v3.2
experiment_freeze.json  registro de congelación; freeze_history/ guarda congelaciones anteriores
scripts/                etapas 01–06 y common.sh
tests/                  128 pruebas
docs/                   documentación (incluye el plan de reorganización diferido)
legacy/                 material histórico: cGAN antigua, bot antiguo, runs.csv, sesiones v2.1
notebooks/              cuaderno exploratorio histórico
human_sessions/         sesiones humanas nuevas (se crea al jugar)
```

Los módulos activos permanecen en la raíz, a propósito, hasta terminar la evaluación final. El plan
de migración a paquetes (`game/`, `simulation/`, `ml/`, `evaluation/`, …) está en
[`docs/REORGANIZACION_POST_EXPERIMENTO.md`](docs/REORGANIZACION_POST_EXPERIMENTO.md).

## 24. Reproducibilidad

- Toda la configuración experimental está versionada y su *hash* se guarda en cada dataset,
  manifiesto y checkpoint. Los datos o modelos incompatibles se rechazan.
- La recolección es determinista por semilla e independiente del número de *workers*.
- Cada etapa registra el commit de git y los *hashes* de configuración.
- Los datos generados, los checkpoints y los resultados no se versionan en git, salvo los registros
  de congelación.

## 25. Uso de herramientas de IA en el desarrollo

Durante el desarrollo se utilizaron herramientas de inteligencia artificial generativa como apoyo
para la revisión de código, la propuesta de refactorizaciones, la generación inicial de pruebas, la
documentación técnica y la revisión lingüística. Las decisiones sobre el problema de investigación,
el alcance del sistema, el diseño experimental, las métricas, los criterios de aceptación y la
interpretación de los resultados fueron tomadas por el autor. Todo el código y el contenido
incorporados fueron revisados, ejecutados y validados por el autor, quien conserva la
responsabilidad sobre la exactitud del informe y el funcionamiento del prototipo.

## 26. Autoría, propósito académico y licencia

- **Autor:** DiegoPerdomo0805.
- **Propósito:** trabajo de graduación sobre ajuste dinámico de dificultad guiado por resultados.
- **Licencia:** este repositorio **todavía no declara una licencia**.
