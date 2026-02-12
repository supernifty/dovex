#!/usr/bin/env python
"""
WSGI entry point for production deployments.
"""
from main import app

if __name__ == "__main__":
    app.run()
