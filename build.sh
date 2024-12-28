#!/usr/bin/env bash

set -o errexit  # Exit immediately if a command exits with a non-zero status

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Run database migrations
flask db migrate -m "migrate 2" || echo "Migration already exists."
flask db upgrade
