# DATA_AUGMENTATION — DEPRECATED

This directory contains the **legacy** bot (`bot_runner.py`) and batch driver
(`bot_swarm.bat`) that generated the old run-level `runs.csv` (6-dimensional,
archetype-labelled schema). They are **not** the data generator for the final
experiment and both now refuse to run (`ROGUE_ALLOW_LEGACY=1` overrides the
Python guard for historical inspection only).

The authoritative generator is the repository-root `bot_runner.py`, which emits
the per-room `RoomSample` contract defined in `telemetry.py`. See the root
`README.md` for the active pipeline.
