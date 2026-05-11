# Agent Notes

- Prefer the local `.venv`, created by `uv`, for all Python commands in this repo.
- Use the `.venv` Python directly instead of the system Python or `uv run`.
- Use `uv pip install` for installing packages.
- Run pre-commit from the local environment for checks, especially Ruff and mypy:
  `.venv/bin/pre-commit run ruff --all-files` and `.venv/bin/pre-commit run mypy --all-files`.
- Do not replace the virtual environment.
