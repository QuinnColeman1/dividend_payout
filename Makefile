# Default target
.DEFAULT_GOAL := help

POETRY := poetry run
PYTHON := poetry run python

format: ## Format code with Ruff
	@echo "🛠️  Formatting code..."
	$(POETRY) ruff check --fix --exit-zero app/ src/

lint-fix: ## Automatically fix linting issues where possible
	@echo "🔧 Fixing linting issues..."
	$(POETRY) ruff check --fix --exit-zero app/ src/

lint: ## Run all linting checks
	@echo "🔍 Running linting checks..."
	$(POETRY) ruff check app/ src/

check-types: ## Run static type checking with mypy
	@echo "🔎 Running type checks..."
	$(POETRY) mypy app/ src/

update: ## Update all dependencies
	poetry update --no-cache

security: ## Run security
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

all: format lint-fix check-types update security clean ## Run all checks and cleanup
	@echo "✅ All checks passed successfully! 🎉"
