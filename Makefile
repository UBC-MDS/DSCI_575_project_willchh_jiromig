.PHONY: setup app test lint format

setup:
	python -m venv .venv
	.venv/Scripts/pip install -r requirements.txt
	pre-commit install
	@if not exist .env copy .env.example .env

app:
	streamlit run app/app.py

test:
	pytest tests/ --cov=src -v

lint:
	pre-commit run --all-files

format:
	ruff check .
