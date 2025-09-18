#!/bin/bash
# MARS Setup Script - Modern Python Environment

set -e  # Exit on any error

echo "🚀 Setting up MARS: Multi-Agent Review System"

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
min_version="3.9"

if [ "$(printf '%s\n' "$min_version" "$python_version" | sort -V | head -n1)" != "$min_version" ]; then
    echo "❌ Python $python_version is too old. Please install Python $min_version or newer."
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies (optional)
if [ "$1" = "--dev" ]; then
    echo "🛠️ Installing development dependencies..."
    pip install pytest black ruff mypy pre-commit
fi

echo "🎯 Setting up Ollama models..."
echo "📋 Make sure you have Ollama installed and running, then run:"
echo "   ollama run llama3.2"
echo "   ollama run mistral"
echo "   ollama run qwen2.5"
echo "   ollama run deepseek-r1"

echo ""
echo "✨ Setup complete!"
echo "🚀 To activate the environment: source venv/bin/activate"
echo "🔥 To run MARS: python MARS.py <cfp_url> <paper_file>"