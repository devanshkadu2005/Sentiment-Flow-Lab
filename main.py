"""Compatibility entrypoint for running FastAPI from project root.

Run:
    python -m uvicorn main:app --host 0.0.0.0 --port 8000
"""

from backend.main import app