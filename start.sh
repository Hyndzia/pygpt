#!/usr/bin/env bash

if [ ! -d "venv" ]; then
    echo "venv not found. Run install.sh first."
    exit 1
fi

source venv/bin/activate

echo "Launching PyGPT!"
python main.py

deactivate