#!/bin/bash

# Movie Sentiment Analyzer Startup Script
# This script sets up and starts the Movie Sentiment Analyzer application

set -e  # Exit on any error

echo "🎬 Movie Sentiment Analyzer - Startup Script"
echo "============================================="

# Colors for output
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

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    print_status "Checking pip installation..."
    if command -v pip3 &> /dev/null; then
        print_success "pip3 found"
    else
        print_error "pip3 is not installed. Please install pip3."
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Dependencies installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p models data logs cache
    print_success "Directories created"
}

# Setup environment file
setup_env() {
    print_status "Setting up environment file..."
    if [ ! -f ".env" ]; then
        cp .env.example .env
        print_warning "Environment file created from template. Please edit .env with your Reddit API credentials."
        print_warning "You can get Reddit API credentials from: https://www.reddit.com/prefs/apps"
    else
        print_success "Environment file already exists"
    fi
}

# Train BERT model
train_model() {
    print_status "Checking if BERT model exists..."
    if [ ! -d "models/bert_sentiment_model" ]; then
        print_status "Training BERT model (this may take a while)..."
        python src/model/bert_trainer.py
        print_success "BERT model trained successfully"
    else
        print_success "BERT model already exists"
    fi
}

# Run tests
run_tests() {
    print_status "Running tests..."
    if command -v pytest &> /dev/null; then
        pytest tests/ -v
        print_success "All tests passed"
    else
        print_warning "pytest not found, skipping tests"
    fi
}

# Start the application
start_app() {
    print_status "Starting the application..."
    echo ""
    echo "🚀 Starting Movie Sentiment Analyzer..."
    echo "📱 Web Interface: http://localhost:5000"
    echo "🔗 API Health Check: http://localhost:5000/api/health"
    echo ""
    echo "Press Ctrl+C to stop the application"
    echo ""
    
    # Check if we're in production mode
    if [ "$FLASK_ENV" = "production" ]; then
        print_status "Starting in production mode with Gunicorn..."
        gunicorn --config gunicorn.conf.py app:app
    else
        print_status "Starting in development mode..."
        python app.py
    fi
}

# Main execution
main() {
    echo ""
    print_status "Starting setup process..."
    
    # Check system requirements
    check_python
    check_pip
    
    # Setup environment
    create_venv
    activate_venv
    install_dependencies
    create_directories
    setup_env
    
    # Setup model
    train_model
    
    # Run tests
    if [ "$1" != "--skip-tests" ]; then
        run_tests
    fi
    
    # Start application
    start_app
}

# Handle command line arguments
case "$1" in
    --help|-h)
        echo "Movie Sentiment Analyzer Startup Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h          Show this help message"
        echo "  --skip-tests        Skip running tests"
        echo "  --setup-only        Only setup, don't start the application"
        echo "  --train-only        Only train the model"
        echo ""
        exit 0
        ;;
    --setup-only)
        check_python
        check_pip
        create_venv
        activate_venv
        install_dependencies
        create_directories
        setup_env
        print_success "Setup completed. Run './start.sh' to start the application."
        exit 0
        ;;
    --train-only)
        activate_venv
        train_model
        print_success "Model training completed."
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac