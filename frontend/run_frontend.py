#!/usr/bin/env python3
"""
Frontend Runner
Script to start the PS-06 Streamlit frontend
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the Streamlit frontend"""
    # Get the frontend directory
    frontend_dir = Path(__file__).parent.absolute()

    # Set environment variables
    os.environ['PYTHONPATH'] = str(frontend_dir.parent)

    # Change to frontend directory
    os.chdir(frontend_dir)

    # Run streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port', '8501',
            '--server.address', '0.0.0.0',
            '--server.headless', 'false',
            '--browser.gatherUsageStats', 'false'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down frontend...")
        sys.exit(0)

if __name__ == "__main__":
    main()