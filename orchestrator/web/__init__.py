"""
Web interface for Streamware Orchestrator.

Provides REST API and WebSocket endpoints for voice testing.
"""

from .server import create_app, run_server

__all__ = ["create_app", "run_server"]
