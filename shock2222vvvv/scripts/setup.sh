#!/bin/bash
# Shock2 AI System - Setup Script

set -e

echo "🚀 Setting up Shock2 AI News System..."
echo "======================================"

# Check Python version
echo "🐍 Checking Python version..."
python3 --version || {
    echo "❌ Python 3.8+ is required"
    exit 1
}

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo "🧠 Downloading spaCy language model..."
python -m spacy download en_core_web_sm

# Create directories
echo "📁 Creating directories..."
mkdir -p data/databases data/models logs output/articles output/reports config

# Set permissions
echo "🔐 Setting permissions..."
chmod +x scripts/*.sh main.py

# Copy default config if not exists
if [ ! -f config.yaml ]; then
    echo "⚙️ Creating default configuration..."
    cp config.yaml.example config.yaml 2>/dev/null || echo "Using default config.yaml"
fi

# Initialize database
echo "🗄️ Initializing database..."
python -c "
from shock2.database.setup import initialize_database
initialize_database()
print('✅ Database initialized')
"

echo ""
echo "✅ Shock2 AI System setup complete!"
echo ""
echo "🎯 Next steps:"
echo "   1. Review configuration: config.yaml"
echo "   2. Set API keys in environment or config"
echo "   3. Run the system: ./scripts/run.sh"
echo ""
echo "📊 Monitoring will be available at: http://localhost:8080"
echo "📁 Generated articles will be in: ./output/"
echo ""
