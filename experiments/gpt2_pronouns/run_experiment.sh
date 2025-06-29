#!/bin/bash
# Run the GPT-2 pronoun context steering experiment

echo "GPT-2 Pronoun Context Steering Experiment"
echo "=========================================="

# Use the virtual environment Python
PYTHON_EXE="../../venv311/Scripts/python.exe"

# Check if Python exists
if [ ! -f "$PYTHON_EXE" ]; then
    echo "Error: Python virtual environment not found at $PYTHON_EXE"
    echo "Using system python instead..."
    PYTHON_EXE="python"
fi

# Run the experiment
echo "Starting experiment..."
$PYTHON_EXE run_experiment.py

echo "Experiment complete!"