#!/bin/bash
# MARS (Multi-Agent Review System) Setup Script
# Modern setup with better error handling and user experience

set -e  # Exit on any error

echo "🚀 Setting up MARS (Multi-Agent Review System)..."

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.10+ is available
check_python() {
    print_status "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python3"
        else
            print_error "Python 3.10+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.10 or higher."
        exit 1
    fi
}

# Create virtual environment
setup_venv() {
    print_status "Setting up virtual environment..."
    
    VENV_NAME="mars-env"
    
    if [ -d "$VENV_NAME" ]; then
        print_warning "Virtual environment '$VENV_NAME' already exists"
        read -p "Do you want to recreate it? (y/n): " recreate
        if [ "$recreate" = "y" ] || [ "$recreate" = "Y" ]; then
            rm -rf "$VENV_NAME"
        else
            print_status "Using existing virtual environment"
        fi
    fi
    
    if [ ! -d "$VENV_NAME" ]; then
        $PYTHON_CMD -m venv "$VENV_NAME"
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    print_success "Virtual environment activated"
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Install dependencies with better error handling
    if pip install -r requirements.txt; then
        print_success "Python dependencies installed"
    else
        print_error "Failed to install Python dependencies"
        print_status "Trying to install core dependencies individually..."
        
        # Core dependencies that are essential
        CORE_DEPS=(
            "ollama>=0.5.0"
            "pypdf>=4.0.0"
            "beautifulsoup4>=4.13.3"
            "requests>=2.32.3"
            "transformers>=4.48.0"
            "torch>=2.0.0"
            "rich>=13.0.0"
            "loguru>=0.7.0"
            "click>=8.1.8"
            "pydantic>=2.10.6"
        )
        
        for dep in "${CORE_DEPS[@]}"; do
            print_status "Installing $dep..."
            pip install "$dep" || print_warning "Failed to install $dep"
        done
    fi
}

# Check and setup Ollama
setup_ollama() {
    print_status "Checking Ollama installation..."
    
    if ! command -v ollama &> /dev/null; then
        print_error "Ollama not found. Please install Ollama first:"
        echo "  Visit: https://ollama.ai"
        echo "  Or run: curl -fsSL https://ollama.ai/install.sh | sh"
        exit 1
    fi
    
    print_success "Ollama found"
    
    # Check if Ollama service is running
    if ! ollama list &> /dev/null; then
        print_warning "Ollama service not running. Please start it:"
        echo "  Run: ollama serve"
        return
    fi
    
    print_status "Setting up AI models..."
    
    # Required models for MARS
    MODELS=("llama3.2" "mistral" "qwen2.5")
    
    for model in "${MODELS[@]}"; do
        print_status "Checking model: $model"
        if ollama list | grep -q "$model"; then
            print_success "Model $model already available"
        else
            print_status "Downloading model: $model (this may take a while...)"
            if ollama pull "$model"; then
                print_success "Model $model downloaded"
            else
                print_warning "Failed to download model $model"
            fi
        fi
    done
}

# Setup development tools (optional)
setup_dev_tools() {
    if [ "$1" = "--dev" ]; then
        print_status "Setting up development tools..."
        
        if [ -f "requirements-dev.txt" ]; then
            pip install -r requirements-dev.txt
            print_success "Development dependencies installed"
        fi
        
        # Setup pre-commit hooks if available
        if command -v pre-commit &> /dev/null; then
            pre-commit install
            print_success "Pre-commit hooks installed"
        fi
    fi
}

# Create necessary directories
setup_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p logs
    mkdir -p data
    mkdir -p outputs
    
    print_success "Directories created"
}

# Main setup process
main() {
    echo "🎯 MARS Setup Starting..."
    echo "=========================="
    
    check_python
    setup_venv
    install_dependencies
    setup_directories
    setup_dev_tools "$1"
    setup_ollama
    
    echo ""
    echo "=========================="
    print_success "🎉 MARS setup completed!"
    echo ""
    echo "To get started:"
    echo "  1. Activate the virtual environment: source mars-env/bin/activate"
    echo "  2. Run MARS: python MARS.py <cfp_url> <paper_path>"
    echo ""
    echo "Example:"
    echo "  python MARS.py https://example.com/cfp paper.pdf"
    echo ""
    print_status "For help: python MARS.py --help"
}

# Run main function with all arguments
main "$@"