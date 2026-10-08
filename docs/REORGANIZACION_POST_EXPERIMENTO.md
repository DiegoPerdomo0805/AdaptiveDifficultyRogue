# Reorganización en paquetes — plan DIFERIDO

> **Estado: DIFERIDO.** Esta migración **no** debe ejecutarse antes de terminar la recolección
> principal y la evaluación final de `exp_v3.2`. Mover los módulos activos cambiaría rutas que
> `freeze_experiment.py verify` compara contra el commit congelado (`CONTRACT_FILES`), además de
> los comandos documentados y las rutas de los scripts. Eso invalidaría la verificación de la
> congelación a mitad del experimento.

## 1. Qué se hizo ya (reorganización segura)

Solo se movió material que ningún código activo importa ni invoca:

| Antes | Después |
|---|---|
| `cGAN.py`, `dummy.py` | `legacy/` |
| `DATA_AUGMENTATION/` | `legacy/data_augmentation/` |
| `runs.csv` | `legacy/data/runs.csv` |
| `logs.txt` | `legacy/logs/logs.txt` |
| `human_sessions/*` (esquema v2.1) | `legacy/human_sessions_v2.1/` |
| `cgan_training_phased.ipynb` | `notebooks/` |

Todo el pipeline activo sigue en la raíz, con las mismas rutas y los mismos comandos.

## 2. Estructura objetivo (después de la evaluación final)

```text
game/            rogue.py, node_map_gen.py
simulation/      dda_core.py, telemetry.py, controllers.py, rewards.py, bot_runner.py,
                 balance.py, loot_dataset.py
ml/              ml_data.py, models.py, model_runtime.py, selection.py,
                 train_q.py, train_cgan.py, train_q_loot.py, train_loot_policy.py
evaluation/      evaluate_controllers.py, analyze_results.py, analyze_loot.py, dataset_tools.py
experiments/     experiment_config.json, experiment_freeze.json, freeze_history/,
                 freeze_experiment.py, salidas por experimento (ignoradas por git)
artifacts/       checkpoints y modelos exportados (ignorados por git; solo manifiestos versionados)
scripts/         01_smoke.sh … 06_analyze.sh, common.sh
tests/
docs/
legacy/
notebooks/
```

Cada directorio de código pasaría a ser un paquete (`__init__.py`), y los comandos se invocarían
como módulos (`python -m simulation.bot_runner ...`) o mediante puntos de entrada de un
`pyproject.toml`.

## 3. Cambios necesarios al migrar

### 3.1 Imports entre módulos (todos son imports planos hoy)

| Módulo | Importa (módulos del repositorio) |
|---|---|
| `analyze_loot` | balance, dda_core, loot_dataset, model_runtime |
| `analyze_results` | balance, dda_core, loot_dataset |
| `balance` | ml_data, models (de forma perezosa, solo en `audit --q-dir`) |
| `bot_runner` | balance, controllers, dda_core, node_map_gen, rewards, telemetry |
| `controllers` | balance, dda_core (+ carga `model_runtime.py` por ruta) |
| `dataset_tools` | balance, dda_core, ml_data, telemetry |
| `evaluate_controllers` | analyze_results, balance, bot_runner, controllers, dda_core, loot_dataset, ml_data, models, telemetry |
| `freeze_experiment` | balance, bot_runner, dda_core, telemetry |
| `loot_dataset` | balance, dda_core, ml_data, telemetry |
| `ml_data` | balance, dda_core, telemetry |
| `model_runtime` | balance, dda_core, ml_data, models, telemetry |
| `models` | balance, dda_core, ml_data |
| `rewards` | balance, controllers, dda_core |
| `rogue` | balance, controllers, dda_core, node_map_gen, rewards, telemetry |
| `selection` | analyze_results, balance, bot_runner, controllers, dda_core, ml_data, models, rewards |
| `telemetry` | balance, dda_core |
| `train_cgan` | balance, bot_runner, dda_core, ml_data, models, selection, telemetry |
| `train_loot_policy` | balance, bot_runner, dda_core, ml_data, model_runtime, models, telemetry, train_q_loot |
| `train_q` | balance, bot_runner, dda_core, ml_data, models, telemetry |
| `train_q_loot` | balance, dda_core, loot_dataset, ml_data, models, telemetry, train_q |

Atención especial: el grafo de dependencias cruza los paquetes propuestos (por ejemplo, `ml`
importa `bot_runner`, y `simulation.balance` importa `ml` de forma perezosa). Hay que importar
con rutas absolutas del paquete y evitar ciclos al inicializar los paquetes.

### 3.2 Código sensible a rutas

- `balance.DEFAULT_CONFIG_PATH`: busca `experiment_config.json` junto a `balance.py`.
- `freeze_experiment.py`: `FREEZE_FILE`, `freeze_history/`, `CONTRACT_FILES` (nombres de archivo
  relativos a la raíz) y `git diff` sobre esas rutas.
- `controllers.load_model_runtime()`: carga `model_runtime.py` por ruta con
  `spec_from_file_location`.
- `model_runtime.py`: inserta su propio directorio en `sys.path`.
- `rogue.py`: `MODEL_DIR` (`model_out`), `LOOT_MODEL_DIR` (`loot_model_out`) y `HUMAN_LOG_DIR`
  (`human_sessions`), todos relativos al archivo.
- `telemetry.git_commit_short()`: ejecuta git en el directorio del módulo.
- `bot_runner`: usa `ProcessPoolExecutor`; los workers deben poder importar el paquete.

### 3.3 Comandos CLI y scripts

- `scripts/01_smoke.sh`: invoca bot_runner, train_q, train_cgan, train_q_loot,
  train_loot_policy, loot_dataset, analyze_loot, evaluate_controllers, analyze_results,
  dataset_tools, balance y freeze_experiment como `"$PY" <archivo>.py`; además, la exclusión de
  compileall menciona `DATA_AUGMENTATION`.
- `scripts/02_collect.sh` … `06_analyze.sh`: `bot_runner.py`, `dataset_tools.py`,
  `loot_dataset.py`, `balance.py`, `freeze_experiment.py`, `train_q.py`, `train_q_loot.py`,
  `train_cgan.py`, `train_loot_policy.py`, `evaluate_controllers.py`, `analyze_results.py` y
  `analyze_loot.py`, junto con `experiment_config.json` leído con `json_get`.
- `scripts/common.sh`: se ejecuta desde la raíz e importa balance, bot_runner, dda_core y
  telemetry.
- README, CLAUDE.md y `docs/`: todos los comandos `python <archivo>.py`.

### 3.4 Pruebas

- `tests/conftest.py` añade la raíz a `sys.path` e importa `bot_runner`.
- Las pruebas importan módulos planos (por ejemplo `import selection` o `import model_runtime`);
  `test_equipment.py` inspecciona el código fuente de `rogue.Player` y
  `bot_runner._simulate_combat_room`.

### 3.5 Congelación

Mover cualquier archivo de `CONTRACT_FILES` hace fallar `freeze_experiment.py verify` contra
`exp_v3.2`. La migración debe hacerse **después** de la evaluación final, como una versión nueva
(por ejemplo `exp_v4.0` o "refactor sin cambio de semántica"), y debe:

1. actualizar `CONTRACT_FILES` a las rutas nuevas;
2. demostrar que los hashes de configuración y de bot (`config_hash`) no cambian, porque no
   dependen de las rutas;
3. reproducir byte a byte una colección de prueba antes y después del cambio
   (`bot_runner --runs N --seed S` con 1 worker);
4. pasar toda la batería de pruebas.

## 4. Orden sugerido (post-experimento)

1. Etiquetar el commit de la evaluación final (`git tag exp_v3.2-final`).
2. Crear la rama `package_refactor`.
3. Mover con `git mv`, añadir los `__init__.py` y reescribir los imports.
4. Actualizar los scripts, la documentación, las pruebas y `CONTRACT_FILES`.
5. Verificar que la salida es idéntica (paso 3.5.3) y que las pruebas pasan.
