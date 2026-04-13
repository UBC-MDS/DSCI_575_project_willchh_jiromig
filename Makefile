.PHONY: setup app test lint format

setup:
# 	python -m venv .venv
# 	.venv/Scripts/pip install -r requirements.txt
	@test -f .env || cp .env.example .env

app:
	streamlit run app/app.py

test:
	pytest tests/ --cov=src -v

lint:
	pre-commit run --all-files

format:
	ruff check .
