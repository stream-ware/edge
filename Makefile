# Streamware - Makefile
# Unified build system for Docker/Enterprise and Embedded/Standalone modes

.PHONY: help install run test test-cov lint clean docker-dev docker-prod docker-stop benchmark models

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
	@if [ ! -d "venv" ]; then python3 -m venv venv; fi
	@. venv/bin/activate && pip install --upgrade pip wheel
	@. venv/bin/activate && pip install -r requirements.txt
	@echo "✅ Installation complete!"
	@echo "   Activate venv: source venv/bin/activate"

install-dev: install
	@. venv/bin/activate && pip install pytest-cov black flake8 mypy
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
	@. venv/bin/activate && python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
	@echo "✅ Models downloaded"

# ============================================
# RUNNING
# ============================================

run:
	@. venv/bin/activate && python -m orchestrator.main

run-embedded:
	@. venv/bin/activate && python -m orchestrator.main config/config-embedded.yaml

run-docker:
	@. venv/bin/activate && python -m orchestrator.main config/config.yaml

# ============================================
# DOCKER
# ============================================

docker-dev:
	docker-compose -f docker-compose-single.yml up --build

docker-prod:
	docker-compose -f docker-compose-full.yml up -d --build

docker-multi:
	docker-compose -f docker-compose-multi.yml up --build

docker-stop:
	docker-compose -f docker-compose-single.yml down
	docker-compose -f docker-compose-full.yml down
	docker-compose -f docker-compose-multi.yml down

docker-logs:
	docker-compose -f docker-compose-single.yml logs -f

docker-build:
	docker build -t streamware-orchestrator .

# ============================================
# TESTING
# ============================================

test:
	@. venv/bin/activate && pytest tests/ -v

test-cov:
	@. venv/bin/activate && pytest tests/ -v --cov=orchestrator --cov-report=html --cov-report=term

test-unit:
	@. venv/bin/activate && pytest tests/ -v -m "not integration"

test-integration:
	@. venv/bin/activate && pytest tests/ -v -m "integration"

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
