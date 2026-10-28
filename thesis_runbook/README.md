# Thesis execution wrapper (cGAN_vs_heuristic)

These scripts **orchestrate**, not modify, the existing `scripts/01_smoke.sh` through `scripts/06_analyze.sh` in AdaptiveDifficultyRogue. Prepared against the publicly visible `cGAN_vs_heuristic` branch on 2026-10-21, not executed on the user's WSL installation.

## Installation
Copy the whole `thesis_runbook` folder to `~/AdaptiveDifficultyRogue/thesis_runbook/`. In WSL:

```bash
cd ~/AdaptiveDifficultyRogue
git switch cGAN_vs_heuristic
chmod +x thesis_runbook/*.sh
./thesis_runbook/00_preflight.sh
```

Before running, check `git status`, git commit and frozen hashes. If the repo branch has advanced since this package was authored, inspect the six stage wrappers before proceeding.

## Execute full frozen run

```bash
cd ~/AdaptiveDifficultyRogue
EXP=experiments/20261021_main RUNS=9000 WORKERS=6 Q_SEEDS='0 1 2' SEEDS='0 1 2' BATCHES='64 128' EVAL_RUNS=400 DEVICE=cuda ./thesis_runbook/10_pipeline.sh
```

### Keeping process running across SSH/terminal disconnects

```bash
sudo apt install tmux # if needed
tmux new -s tesis
./thesis_runbook/10_pipeline.sh
# detach: Ctrl+B then D; reattach: tmux attach -t tesis
```

Run on a Linux/WSL filesystem and avoid Windows sleep/hibernation during collection/training. Check GPU/CPU temperatures and remaining storage. Do not run a second pipeline against the same EXP concurrently.

## Progress / archive

```bash
./thesis_runbook/20_status.sh
./thesis_runbook/30_archive.sh
```

## Safe failure behavior
- Failed stages retain their directory and logs. They cannot be automatically overwritten; this is intentional, because a partially collected dataset is evidence, not junk.
- Inspect the failing stage's `log.txt` and `meta.json` before determining if a clean restart on a new experiment directory is appropriate. Do not hand-create DONE markers.
- Existing DONE-marked stages will be skipped. Verify that experiment identity and settings match before resuming.
- A FAILED Q gate blocks final cGAN; a FAILED loot gate is recorded without blocking enemy-axis evaluation.
- The repository's Q stage currently selects **the first listed Q seed** as q_selected after all three gates pass; if methodological selection criteria require the best seed, this needs an explicitly reviewed code change and a new frozen experimental version, not a silent script adjustment.
- Script `04_train_cgan.sh` defaults to **200 epochs** per candidate; with 3 seeds and 2 batch sizes this can dominate the timeline.

## Experimental outcomes
Artifacts should appear under `experiments/20261021_main/06_analyze/results/`, with provenance and logs in each stage directory. A stage's DONE marker confirms its native validation passed, but **does not guarantee** all dissertation claims are supported; inspect Q metrics, candidate report, OOD, and controller metrics before writing results.

## Runtime note
The 9000-run dataset + 6 candidate GAN sweep may take longer than 12 hours on the target machine. Measure `02_collect/blocks/block_000.csv` completion time and first cGAN training run to update ETA. Do not reduce the frozen 9000-game count or change candidate sweep post hoc and label it the same definitive experiment.
