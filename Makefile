# Default target
.DEFAULT_GOAL := help

POETRY := poetry run
PYTHON := poetry run python

.PHONY: help format lint-fix lint check-types test security update clean all

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

format: ## Format code with Ruff
	@echo "🛠️  Formatting code..."
	$(POETRY) ruff format app/ src/ tests/

lint-fix: ## Automatically fix linting issues where possible
	@echo "🔧 Fixing linting issues..."
	$(POETRY) ruff check --fix --exit-zero app/ src/

lint: ## Run all linting checks
	@echo "🔍 Running linting checks..."
	$(POETRY) ruff check app/ src/ tests/

check-types: ## Run static type checking with mypy
	@echo "🔎 Running type checks..."
	$(POETRY) mypy app/ src/

test: ## Run the test suite with coverage
	@echo "🧪 Running tests..."
	$(POETRY) pytest

update: ## Update all dependencies
	poetry update --no-cache

security: ## Run security checks with bandit and pip-audit
	@echo "🔒 Running security checks..."
	$(POETRY) bandit -r app/ src/
	$(POETRY) pip-audit --ignore-vuln GHSA-4xh5-x5gv-qwph

clean: ## Remove all cache and build files
	find . -type f -name '*.py[co]' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .tox/
	rm -f coverage.xml
	rm -f *.cover
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	find . -name '*.egg-info' -exec rm -rf {} +
	rm -rf .ruff_cache/
	rm -rf .pylint.d/
	find . -name '.DS_Store' -exec rm -f {} \;

all: lint check-types test security ## Run all checks (does not change dependencies)
	@echo "✅ All checks passed successfully! 🎉"
