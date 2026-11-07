PORT ?= 8824

.PHONY: install test lint format train train-quick backtest gate run docker-build docker-run screenshot clean

install:
	uv sync --python 3.12

test:
	uv run pytest -q

lint:
	uvx ruff check .
	uvx ruff format --check .

format:
	uvx ruff format .
	uvx ruff check . --fix

train:  ## full pipeline into data/artifacts.json (about 1 minute)
	uv run demandcast train

train-quick:  ## 420 days of history, 4 origins (about 15 seconds)
	uv run demandcast train --quick

backtest:
	uv run demandcast backtest

gate:  ## what CI runs: fails unless the model beats seasonal naive at the 7-day horizon
	uv run demandcast backtest --quick --gate

run:
	uv run demandcast serve --port $(PORT)

docker-build:
	docker build -t demand-forecast .

docker-run:
	docker run --rm -p $(PORT):8824 demand-forecast

screenshot:  ## needs the server running on $(PORT) and Google Chrome
	uv run python scripts/screenshot.py --port $(PORT)

clean:
	rm -rf .pytest_cache .ruff_cache data
