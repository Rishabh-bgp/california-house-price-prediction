# Contributing
1. `pip install -r requirements-dev.txt`; `python -m src.train && ruff check . && pytest -q` must pass.
2. Keep predictions clipped at the $500k cap.
3. Re-execute the notebook if model settings change, so the docs stay accurate. MIT licence applies.
