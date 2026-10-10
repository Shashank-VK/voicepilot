.PHONY: setup dev test lint format bench demo clean

PYTHON ?= python3
VENV ?= .venv
BIN ?= $(VENV)/bin

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/pip install -e .
	@echo "Setup complete. Run 'make dev' to start the application."

dev:
	$(BIN)/uvicorn voicepilot.server.app:app --host 0.0.0.0 --port 8000 --reload

test:
	$(BIN)/pytest -v --cov=voicepilot --cov-report=term-missing

lint:
	$(BIN)/ruff check .

format:
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

bench:
	$(BIN)/python -m voicepilot.cli bench --iterations 10

demo:
	@echo "Starting VoicePilot in Demo Mode on http://localhost:8000..."
	$(BIN)/python -m voicepilot.cli serve --host 0.0.0.0 --port 8000

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov dist build *.egg-info
