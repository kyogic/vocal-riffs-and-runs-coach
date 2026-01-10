#!/bin/bash
# Setup script for Vocal Riffs and Runs Coach

echo "====================================="
echo "Vocal Riffs and Runs Coach - Setup"
echo "====================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "====================================="
echo "Setup complete!"
echo "====================================="
echo ""
echo "To run the application:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run the application: python main.py"
echo ""
echo "For Windows users:"
echo "  1. Activate: venv\\Scripts\\activate"
echo "  2. Run: python main.py"
echo ""
