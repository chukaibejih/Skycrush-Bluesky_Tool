
#!/usr/bin/env bash

set -o errexit  # Exit immediately if a command exits with a non-zero status

# Upgrade pip (uncomment if necessary)
# pip install --upgrade pip

# Install dependencies from requirements.txt
pip install -r requirements.txt

flask db init

flask db migrate -m "Initial migration"

flask db upgrade
