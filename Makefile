.PHONY: all venv install format lint security check prepare_data train_model test clean

# 1. Create virtual environment
venv:
	python3 -m venv venv

# 2. Install dependencies
install: venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

# 3. Code formatting
format:
	. venv/bin/activate && black .

# 4. Linting
lint:
	. venv/bin/activate && flake8 .

# 5. Security check
security:
	. venv/bin/activate && pip install bandit && bandit -r .

# 6. Run all code checks
check: format lint security

# 7. Prepare data
prepare_data:
	. venv/bin/activate && python3 -c "from main import prepare_data; prepare_data()"

# 8. Train model
train_model:
	. venv/bin/activate && python3 -c "from main import train_model; train_model()"

# 9. Run tests
test:
	. venv/bin/activate && pytest

# 10. Clean up
clean:
	rm -rf venv __pycache__ .pytest_cache .mypy_cache

app:
	python3 main.py	

up:
	docker compose up --build

down:
	docker compose down

# 11. Run everything
all: install check prepare_data train_model test

