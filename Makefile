# Makefile for XQC development automation

.PHONY: docs clean test lint

test:
	pytest -v

lint:
	ruff check xqc/

docs:
	@echo "Generating documentation..."
	python docs/generate_docs.py

clean:
	@echo "Cleaning build artifacts..."
	rm -rf docs/_build build dist *.egg-info
