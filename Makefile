.PHONY: setup download-data build-corpus build-indices app test lint format

setup: download-data build-corpus build-indices

download-data:
	@echo "==> Downloading raw data..."
	@python -u -c "from src.utils import download_raw_data; download_raw_data('data/raw')"

build-corpus:
	@echo "==> Building processed corpus..."
	@python -u -c "from src.utils import build_processed_corpus; build_processed_corpus('data/raw', 'data/processed')"

build-indices:
	@echo "==> Building search indices..."
	@python -u -c "from src.utils import build_indices; build_indices('data/processed/product_corpus.parquet', 'indices')"

app:
	streamlit run app/app.py

test:
	pytest tests/ --cov=src --cov-fail-under=90 --cov-report=term-missing -v

lint:
	pre-commit run --all-files

format:
	ruff check .
