#!/bin/bash

echo "Setting up NESCAC Swimming Analytics Web Application..."

# Get the project root directory (parent of webapp)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# Check if virtual environment exists
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "Error: Virtual environment not found at $PROJECT_ROOT/.venv"
    echo "Please create a virtual environment in the project root first:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Install additional backend dependencies
echo "Installing additional backend dependencies..."
cd backend
source ../../.venv/bin/activate
pip install -r requirements.txt
cd ..

# Setup frontend
echo "Setting up React frontend..."
cd frontend
npm install
cd ..

echo "Setup complete!"
echo ""
echo "To run the application:"
echo "1. Activate your virtual environment: source .venv/bin/activate"
echo "2. Start the backend: cd webapp/backend && python app.py"
echo "3. Start the frontend: cd webapp/frontend && npm start"
echo ""
echo "The application will be available at:"
echo "- Frontend: http://localhost:3000"
echo "- Backend API: http://localhost:5000" 