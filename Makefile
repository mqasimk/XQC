# Makefile for XQC

.PHONY: install test lint format docs clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

lint:
	ruff check xqc/

format:
	ruff format xqc/

docs:
	@echo "Generating documentation..."
	python docs/generate_docs.py

clean:
	@echo "Cleaning build artifacts..."
	rm -rf docs/_build build dist *.egg-info
