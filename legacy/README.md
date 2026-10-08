# legacy/ — material histórico (no usar en el experimento)

Este directorio conserva, sin borrarlos, archivos de etapas anteriores del proyecto. **Ninguno forma
parte del pipeline activo** ni es importado o invocado por él. Se movieron aquí (con `git mv`, para
conservar el historial) durante la reorganización del repositorio posterior a la congelación de
`exp_v3.2`.

| Ruta | Origen | Motivo por el que es histórico |
|---|---|---|
| `cGAN.py` | raíz | cGAN antigua de 6 variables por corrida, entrenada para **reconstruir** acciones heurísticas. Termina con error al ejecutarse o importarse (`ROGUE_ALLOW_LEGACY=1` lo permite solo para consulta). |
| `dummy.py` | raíz | Script exploratorio (CUDA/pandas). También termina con error. |
| `data_augmentation/` | `DATA_AUGMENTATION/` | Bot antiguo que generaba `runs.csv` y su lanzador `bot_swarm.bat`; ambos están deshabilitados. |
| `data/runs.csv` | raíz | Datos antiguos por corrida (esquema de 6 dimensiones), incompatibles con `room_samples_v4.0`. |
| `logs/logs.txt` | raíz | Registro de consola pegado de una ejecución antigua. |
| `human_sessions_v2.1/` | `human_sessions/` | Sesiones humanas grabadas con el esquema `room_samples_v2.1`; el validador actual las rechaza. Las sesiones nuevas se siguen escribiendo en `human_sessions/` (lo crea `rogue.py`). |

Las rutas internas de estos scripts (por ejemplo `runs.csv` relativo) no se actualizaron: los
scripts no deben ejecutarse. El cuaderno exploratorio está en `notebooks/`.
