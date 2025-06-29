#!/bin/bash
# Shock2 AI System - Run Script

set -e

echo "🚀 Starting Shock2 AI News System..."
echo "===================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if setup was run
if [ ! -d "data" ] || [ ! -f "requirements.txt" ]; then
    echo "⚠️ System not set up. Running setup first..."
    ./scripts/setup.sh
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1

# Create log file with timestamp
LOG_FILE="logs/shock2_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "📝 Logging to: $LOG_FILE"
echo "🎯 System starting..."
echo ""

# Run the system
python main.py 2>&1 | tee "$LOG_FILE"
