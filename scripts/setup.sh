#!/bin/bash

# Dance Video Generator - Setup Script

set -e

echo "🎬 Dance Video Generator - Setup"
echo "================================="

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first:"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose:"
    echo "   https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker found"

# Create .env from example if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your API keys if using proprietary services"
fi

# Create models directory
echo "📁 Creating models directory..."
mkdir -p models/{fomm,liveportrait,magic-animate,magic-dance,sam}

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected!"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "   GPU acceleration will be enabled"
else
    echo "⚠️  No NVIDIA GPU detected. Will run on CPU (slower)"
    echo "   For GPU support, install NVIDIA Docker: https://github.com/NVIDIA/nvidia-docker"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download AI models:"
echo "   python scripts/download_models.py"
echo ""
echo "2. Start the application:"
echo "   docker-compose up -d"
echo ""
echo "3. Open in browser:"
echo "   http://localhost:3001"
echo ""
echo "4. Monitor Celery workers:"
echo "   http://localhost:5555"
echo ""
