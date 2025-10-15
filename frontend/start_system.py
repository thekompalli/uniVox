#!/usr/bin/env python3
"""
PS-06 System Launcher
Script to start both backend and frontend services
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path
import requests

class PS06SystemLauncher:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.absolute()
        self.backend_process = None
        self.frontend_process = None

    def check_backend_health(self, max_attempts=30):
        """Check if backend is healthy"""
        print("Waiting for backend to start...")

        for attempt in range(max_attempts):
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    print("✅ Backend is ready!")
                    return True
            except requests.RequestException:
                pass

            time.sleep(2)
            print(f"Attempt {attempt + 1}/{max_attempts} - Backend not ready yet...")

        print("❌ Backend failed to start in time")
        return False

    def start_backend(self):
        """Start the FastAPI backend"""
        print("🚀 Starting PS-06 Backend...")

        backend_dir = self.project_root / "src"

        try:
            self.backend_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn",
                "main:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ], cwd=backend_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            print("Backend process started with PID:", self.backend_process.pid)
            return True

        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False

    def start_frontend(self):
        """Start the Streamlit frontend"""
        print("🎨 Starting PS-06 Frontend...")

        frontend_dir = self.project_root / "frontend"

        try:
            self.frontend_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", "app.py",
                "--server.port", "8501",
                "--server.address", "0.0.0.0",
                "--server.headless", "false",
                "--browser.gatherUsageStats", "false"
            ], cwd=frontend_dir)

            print("Frontend process started with PID:", self.frontend_process.pid)
            return True

        except Exception as e:
            print(f"❌ Failed to start frontend: {e}")
            return False

    def shutdown(self):
        """Shutdown both services"""
        print("\n🔄 Shutting down PS-06 System...")

        if self.frontend_process:
            print("Stopping frontend...")
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()

        if self.backend_process:
            print("Stopping backend...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()

        print("✅ System shutdown complete")

    def run(self):
        """Run the complete PS-06 system"""
        print("=" * 50)
        print("🎯 PS-06 System Launcher")
        print("Language Agnostic Speaker ID & Diarization")
        print("=" * 50)

        try:
            # Start backend
            if not self.start_backend():
                return

            # Wait for backend to be ready
            if not self.check_backend_health():
                self.shutdown()
                return

            # Start frontend
            if not self.start_frontend():
                self.shutdown()
                return

            print("\n" + "=" * 50)
            print("✅ PS-06 System is now running!")
            print("📊 Backend API: http://localhost:8000")
            print("🎨 Frontend UI: http://localhost:8501")
            print("📚 API Docs: http://localhost:8000/docs")
            print("=" * 50)
            print("\nPress Ctrl+C to shutdown both services...")

            # Keep the launcher running
            while True:
                time.sleep(1)

                # Check if processes are still alive
                if self.backend_process.poll() is not None:
                    print("❌ Backend process died unexpectedly")
                    break

                if self.frontend_process.poll() is not None:
                    print("❌ Frontend process died unexpectedly")
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

def main():
    launcher = PS06SystemLauncher()
    launcher.run()

if __name__ == "__main__":
    main()