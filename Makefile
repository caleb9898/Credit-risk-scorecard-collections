.PHONY: install lint format test check notebooks data-check clean

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest

# The gate: nothing is committed or reported unless this passes.
check: lint test

# Execute every notebook top-to-bottom on a clean kernel (reproducibility check).
notebooks:
	@for nb in $$(ls notebooks/*.ipynb 2>/dev/null); do \
	  echo "Executing $$nb"; \
	  jupyter nbconvert --to notebook --execute --inplace \
	    --ExecutePreprocessor.record_timing=False $$nb || exit 1; \
	done

# Validate the raw data file against the expected schema and print its SHA-256.
data-check:
	python -m creditrisk.data.schema data/raw/$(FILE)

clean:
	rm -rf .pytest_cache .ruff_cache build *.egg-info
