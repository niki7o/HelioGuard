# HelioGuard — common operations
# Run `make help` to see available targets.

PYTHON ?= python3
PYTHONPATH := src

.PHONY: help install data test lint nb-audit clean

help:
	@echo "HelioGuard targets:"
	@echo "  install     pip install -r requirements.txt"
	@echo "  data        download raw OMNI + NCEI data (idempotent)"
	@echo "  data-audit  build + execute the Day-1 audit notebook"
	@echo "  test        run pytest"
	@echo "  lint        ruff check ."
	@echo "  mlflow-ui   launch MLflow UI on http://localhost:5000"
	@echo "  clean       remove __pycache__, .pytest_cache, .ipynb_checkpoints"

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m helioguard.data.download

data-audit:
	$(PYTHON) scripts/build_notebook_00.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m jupyter nbconvert \
		--to notebook --execute --inplace notebooks/00_data_audit.ipynb

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/ -v

lint:
	ruff check src/ tests/ scripts/

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri ./mlruns --port 5000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
	find . -name '.ipynb_checkpoints' -exec rm -rf {} + 2>/dev/null || true
