# Makefile for XQC documentation automation

.PHONY: docs clean

docs:
	@echo "Generating documentation..."
	python docs/generate_docs.py

clean:
	@echo "Cleaning generated documentation..."
	rm -rf docs/_build
