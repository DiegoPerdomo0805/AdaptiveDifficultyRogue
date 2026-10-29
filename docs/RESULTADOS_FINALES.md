# Resultados finales de la tesis

## Resultados experimentales finales

### Alcance

El proyecto estudia el ajuste dinámico de dificultad mediante
generación condicional de configuraciones de combate en un videojuego
roguelike experimental desarrollado en Python.

El experimento principal `exp_v3.2` conserva su estado congelado.
El seguimiento híbrido constituye una evaluación exploratoria
independiente y no modifica los resultados originales.

### Controlador híbrido seleccionado

- Factor de mezcla: `alpha = 0.50`
- Candidato seleccionado: `closest_ood`
- Checkpoint: `273ce5c8f782dcdf`
- Semillas pareadas: 393
- Métrica comparativa: distancia de balance por partida, B_r
- Interpretación: valores menores representan menor distancia al objetivo.

### Comparación pareada

| Comparación | Diferencia media de B_r | IC bootstrap 95 % |
|---|---:|---|
| Híbrido - fijo | -0.148779 | [-0.162513, -0.135445] |
| Híbrido - heurístico | -0.057633 | [-0.072029, -0.043545] |

Ambos intervalos están por debajo de cero. En las partidas
comparables, el controlador híbrido presentó una distancia media
de balance inferior a ambas referencias.

Estos resultados corresponden a evaluación automatizada.
No demuestran una mejora de la experiencia subjetiva humana.

### Evidencia reproducible

- `experiments/20261022_hybrid_v1/plan.json`
- `experiments/20261022_hybrid_v1/validation.json`
- `experiments/20261022_hybrid_v1/selection.lock.json`
- `experiments/20261022_hybrid_v1/final/final_report.json`
- `scripts/thesis_hybrid_recovery.sh`
- `thesis_runbook/`
- `experiment_freeze.json`

### Verificación

```bash
python freeze_experiment.py verify
pytest -q
```

### Limitaciones

La evaluación corresponde a un seguimiento exploratorio con
semillas automatizadas. La evidencia no demuestra eficacia
generalizada sobre jugadores humanos.

El controlador híbrido estudiado no equivale a una cGAN
autónoma. La selección se realizó mediante una fase previa
independiente de la evaluación final.

Los artefactos experimentales históricos se preservan sin
volver a ejecutar las fases de selección ni evaluación final.
