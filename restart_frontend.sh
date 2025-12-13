#!/bin/bash

# Script to restart the frontend server

echo "=========================================="
echo "🔄 Restarting Frontend Server"
echo "=========================================="

# Navigate to frontend directory
cd "$(dirname "$0")/frontend" || exit 1

# Start the frontend
echo "🚀 Starting frontend server on http://localhost:3000..."
echo "=========================================="
echo ""

npm run dev
