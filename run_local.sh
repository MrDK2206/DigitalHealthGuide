#!/bin/bash
# macOS/Linux local development script

echo "📦 Installing/updating dependencies..."
python3 -m pip install -q -r requirements.txt

echo "✅ Dependencies ready!"
echo "🚀 Starting Flask development server..."
echo "   Visit: http://127.0.0.1:5000"
echo ""

python3 app.py
