.PHONY: setup download-data build-corpus app test lint format

REPO := UBC-MDS/DSCI_575_project_willchh_jiromig
TAG  := v0.1.0

setup: download-data build-corpus
	@test -f .env || cp .env.example .env
	@if [ ! -f indices/bm25_index.pkl ]; then \
		echo "Downloading pre-built indices from GitHub Release $(TAG)..."; \
		mkdir -p indices/faiss_index; \
		gh release download $(TAG) --repo $(REPO) --pattern "bm25_index.pkl" --dir indices; \
		gh release download $(TAG) --repo $(REPO) --pattern "index.faiss" --dir indices/faiss_index; \
		gh release download $(TAG) --repo $(REPO) --pattern "corpus.pkl" --dir indices/faiss_index; \
		echo "Indices downloaded successfully."; \
	else \
		echo "Indices already exist, skipping download."; \
	fi

download-data:
	@python -c "from src.utils import download_raw_data; download_raw_data('data/raw')"

build-corpus:
	@python -c "from src.utils import build_processed_corpus; build_processed_corpus('data/raw', 'data/processed')"

app:
	streamlit run app/app.py

test:
	pytest tests/ --cov=src -v

lint:
	pre-commit run --all-files

format:
	ruff check .
