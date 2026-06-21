# HelioGuard - common operations
# Run `make help` to see available targets.

PYTHON ?= python3
PYTHONPATH := src

.PHONY: help install data notebook test lint mlflow-ui clean

help:
	@echo "HelioGuard targets:"
	@echo "  install     pip install -r requirements.txt"
	@echo "  data        download raw OMNI + NCEI data (idempotent)"
	@echo "  notebook    rebuild and execute notebooks/helioguard.ipynb"
	@echo "  test        run pytest"
	@echo "  lint        ruff check ."
	@echo "  mlflow-ui   launch MLflow UI on http://localhost:5000"
	@echo "  clean       remove __pycache__, .pytest_cache, .ipynb_checkpoints"

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m helioguard.data.download

notebook:
	$(PYTHON) scripts/build_notebook.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m nbconvert \
		--to notebook --execute --inplace notebooks/helioguard.ipynb \
		--ExecutePreprocessor.timeout=1200

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/ -v

lint:
	ruff check src/ tests/ scripts/

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
	find . -name '.ipynb_checkpoints' -exec rm -rf {} + 2>/dev/null || true
