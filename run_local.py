#!/usr/bin/env python
"""
Cross-platform local development runner.
Works on Windows, macOS, and Linux without hardcoded paths.
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)
    
    # Ensure dependencies are installed
    print("📦 Installing/updating dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        check=True
    )
    
    print("✅ Dependencies ready!")
    print("🚀 Starting Flask development server...")
    print("   Visit: http://127.0.0.1:5000\n")
    
    # Run Flask app
    subprocess.run(
        [sys.executable, "app.py"],
        cwd=project_root
    )

if __name__ == "__main__":
    main()
