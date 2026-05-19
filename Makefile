.PHONY: help clean clean-build clean-pyc clean-test test benchmark-smoke install
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
    match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
    if match:
        target, help = match.groups()
        print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove local build, bytecode, and test artifacts

clean-build: ## remove package build artifacts
	rm -rf build/ dist/ .eggs/
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python bytecode artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

clean-test: ## remove local test artifacts
	rm -rf .tox/ htmlcov/ .pytest_cache/ tests/.tmp/
	rm -f .coverage

test: ## run the LG-FGT focused tests
	pytest tests/test_feature_gate_transformer.py

benchmark-smoke: ## run a tiny benchmark smoke test
	python benchmarks/run_benchmark.py --dataset synthetic_classification --models logistic_regression feature_gate_transformer_adaptive --sample-size 300 --max-epochs 1 --accelerator cpu --output-dir .pt_tmp/benchmark_smoke

install: ## install this fork in editable mode with dev extras
	pip install -e .[dev]
