@echo off
REM DEPRECATED: drives the legacy DATA_AUGMENTATION/bot_runner.py (old runs.csv schema).
REM Use the root pipeline in README.md instead (scripts/02_collect.sh).
echo DEPRECATED: use the root bot_runner.py pipeline described in README.md 1>&2
exit /b 1
REM Original command (kept for the record):
REM for /l %%x in (1,1,100) do python bot_runner.py --runs 200 --seed %%x
