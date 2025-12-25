# Streamware - Makefile
# Unified build system for Docker/Enterprise and Embedded/Standalone modes

.PHONY: help install run run-web test test-cov lint clean docker-dev docker-prod docker-stop benchmark models stop stop-local venv-reset reinstall

COMPOSE_SINGLE = docker compose -p streamware-single -f docker-compose-single.yml
COMPOSE_FULL   = docker compose -p streamware-full -f docker-compose-full.yml
COMPOSE_MULTI  = docker compose -p streamware-multi -f docker-compose-multi.yml

PYTHON ?= /usr/bin/python3
VENV_PY = venv/bin/python

# Default target
help:
	@echo "╔═══════════════════════════════════════════════════════════╗"
	@echo "║  Streamware - Build Commands                              ║"
	@echo "╚═══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  Installation:"
	@echo "    make install        - Install dependencies (venv + pip)"
	@echo "    make install-dev    - Install with dev dependencies"
	@echo "    make models         - Download AI models (Piper, YOLO)"
	@echo ""
	@echo "  Running:"
	@echo "    make run            - Run orchestrator (native Python)"
	@echo "    make run-embedded   - Run in embedded mode"
	@echo "    make run-docker     - Run in docker mode"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-dev     - Start development stack"
	@echo "    make docker-prod    - Start production stack"
	@echo "    make docker-stop    - Stop all containers"
	@echo "    make docker-logs    - Show container logs"
	@echo ""
	@echo "  Testing:"
	@echo "    make test           - Run all tests"
	@echo "    make test-cov       - Run tests with coverage"
	@echo "    make test-unit      - Run unit tests only"
	@echo ""
	@echo "  Utilities:"
	@echo "    make benchmark      - Run performance benchmark"
	@echo "    make lint           - Run code linting"
	@echo "    make clean          - Clean build artifacts"
	@echo ""

# ============================================
# INSTALLATION
# ============================================

install:
	@echo "Installing Streamware..."
	@if [ ! -d "venv" ]; then $(PYTHON) -m venv venv; fi
	@$(VENV_PY) -m pip install --upgrade pip wheel
	@$(VENV_PY) -m pip install -r requirements.txt
	@echo "✅ Installation complete!"
	@echo "   Activate venv: source venv/bin/activate"

venv-reset:
	@rm -rf venv

reinstall: venv-reset install

install-dev: install
	@$(VENV_PY) -m pip install pytest-cov black flake8 mypy
	@echo "✅ Dev dependencies installed"

install-jetson:
	@echo "Running Jetson installation script..."
	@chmod +x scripts/install.sh
	@./scripts/install.sh

models:
	@echo "Downloading AI models..."
	@chmod +x scripts/download_piper_pl.sh
	@./scripts/download_piper_pl.sh medium gosia
	@echo "Downloading YOLOv8..."
	@$(VENV_PY) -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
	@echo "✅ Models downloaded"

# ============================================
# RUNNING
# ============================================

run:
	@env -u LD_LIBRARY_PATH -u LD_PRELOAD -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u _CE_CONDA -u _CE_M $(VENV_PY) -m orchestrator.main

run-embedded:
	@env -u LD_LIBRARY_PATH -u LD_PRELOAD -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u _CE_CONDA -u _CE_M $(VENV_PY) -m orchestrator.main config/config-embedded.yaml

run-docker:
	@env -u LD_LIBRARY_PATH -u LD_PRELOAD -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u _CE_CONDA -u _CE_M $(VENV_PY) -m orchestrator.main config/config.yaml

run-web:
	@echo "Starting web interface on http://localhost:8000"
	@env -u LD_LIBRARY_PATH -u LD_PRELOAD -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u _CE_CONDA -u _CE_M $(VENV_PY) -m orchestrator.web.server

# ============================================
# DOCKER (Optimized for fast builds/restarts)
# ============================================

# Build with BuildKit cache (fast rebuilds)
docker-build:
	DOCKER_BUILDKIT=1 docker build \
		--cache-from streamware-orchestrator:latest \
		-t streamware-orchestrator:latest .

# Development (with build cache)
docker-dev:
	DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 \
		$(COMPOSE_SINGLE) up --build --remove-orphans

# Development (no rebuild - fastest restart)
docker-up:
	$(COMPOSE_SINGLE) up --remove-orphans

docker-up-demo:
	COMPOSE_PROFILES=demo $(COMPOSE_SINGLE) up --remove-orphans

# Production (detached, cached build)
docker-prod:
	DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 \
		$(COMPOSE_FULL) up -d --build --remove-orphans

# Production restart (no rebuild)
docker-restart:
	$(COMPOSE_FULL) restart orchestrator

docker-multi:
	DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 \
		$(COMPOSE_MULTI) up --build --remove-orphans

docker-stop:
	$(COMPOSE_SINGLE) down --remove-orphans 2>/dev/null || true
	$(COMPOSE_FULL) down --remove-orphans 2>/dev/null || true
	$(COMPOSE_MULTI) down --remove-orphans 2>/dev/null || true

stop: docker-stop stop-local

stop-local:
	@pkill -f -x "python -m orchestrator.main" 2>/dev/null || true
	@pkill -f -x "python3 -m orchestrator.main" 2>/dev/null || true
	@pkill -f -x "venv/bin/python -m orchestrator.main" 2>/dev/null || true
	@pkill -f -x "venv/bin/python -m orchestrator.main config/config.yaml" 2>/dev/null || true
	@pkill -f -x "venv/bin/python -m orchestrator.main config/config-embedded.yaml" 2>/dev/null || true
	@pkill -f -x "python firmware/sim.py" 2>/dev/null || true
	@pkill -f -x "python3 firmware/sim.py" 2>/dev/null || true
	@pkill -f -x "python scripts/benchmark.py" 2>/dev/null || true
	@pkill -f -x "python3 scripts/benchmark.py" 2>/dev/null || true

docker-logs:
	$(COMPOSE_SINGLE) logs -f orchestrator

# Pull base images (do once, speeds up builds)
docker-pull:
	docker pull python:3.11-slim
	docker pull eclipse-mosquitto:2
	docker pull nginx:alpine

# ============================================
# TESTING
# ============================================

test:
	PYTHONPATH=. python3 -m pytest tests/ -v

test-cov:
	PYTHONPATH=. python3 -m pytest tests/ -v --cov=orchestrator --cov-report=html --cov-report=term

test-unit:
	PYTHONPATH=. python3 -m pytest tests/ -v -m "not integration"

test-integration:
	PYTHONPATH=. python3 -m pytest tests/ -v -m "integration"

# ============================================
# UTILITIES
# ============================================

benchmark:
	@. venv/bin/activate && python scripts/benchmark.py --all

benchmark-vision:
	@. venv/bin/activate && python scripts/benchmark.py --vision

lint:
	@. venv/bin/activate && flake8 orchestrator/ tests/ --max-line-length=100

format:
	@. venv/bin/activate && black orchestrator/ tests/

typecheck:
	@. venv/bin/activate && mypy orchestrator/

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf __pycache__ */__pycache__ */*/__pycache__
	@rm -rf .pytest_cache htmlcov .coverage
	@rm -rf *.egg-info build dist
	@rm -rf venv
	@echo "✅ Cleaned"

clean-docker:
	docker-compose -f docker-compose-single.yml down -v --rmi local
	docker-compose -f docker-compose-full.yml down -v --rmi local

# ============================================
# DEVELOPMENT
# ============================================

dev-setup: install-dev models
	@echo "✅ Development environment ready"

shell:
	@. venv/bin/activate && python

logs:
	@tail -f logs/streamware.log
