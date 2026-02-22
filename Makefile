TARGET_DIRS := src tests

lint:
	@uv run ruff check $(TARGET_DIRS)
	@uv run mypy --strict $(TARGET_DIRS)

format:
	@uv run ruff check --fix $(TARGET_DIRS)

test:
	@uv run pytest --cov

PRESET ?= xs

bench:
	@uv run python -m pumpference.benchmark --preset $(PRESET)
