.PHONY: fix check docs docs-serve
PYTHON ?= python
SOURCES = leitwerk tests examples benchmarks

fix:
	$(PYTHON) -m ruff check --fix --unsafe-fixes $(SOURCES)
	$(PYTHON) -m ruff format $(SOURCES)

check:
	$(PYTHON) -m ruff check $(SOURCES)
	$(PYTHON) -m ruff format --check $(SOURCES)
	$(PYTHON) -m mypy $(SOURCES)
	$(PYTHON) -m pytest -q

docs:
	$(PYTHON) -m mkdocs build

docs-serve:
	$(PYTHON) -m mkdocs serve
