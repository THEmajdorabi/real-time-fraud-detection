.PHONY: setup train serve test lint benchmark docker

setup:
	uv sync --extra dev

train:
	uv run python scripts/train.py --rows 50000

serve:
	uv run uvicorn fraud_detection.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	uv run pytest

lint:
	uv run ruff check .

benchmark:
	uv run python scripts/benchmark.py

docker:
	docker compose up --build

