#!/usr/bin/env python3
"""
Simple script to start the FastAPI server with proper configuration.
"""

import uvicorn
import sys
import os


def main():
    print("🎤 Starting Sentiotech Emotion Recognition Server...")
    print("=" * 50)

    # Check if required files exist
    required_files = ["back.py", "inference.py", "models.py", "utils.py"]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Make sure you're running this from the correct directory.")
        sys.exit(1)

    print("✅ All required files found")
    print("🚀 Starting server on http://localhost:8000")
    print("📖 API docs available at http://localhost:8000/docs")
    print("❤️  Health check at http://localhost:8000/health")
    print("=" * 50)
    print("Press Ctrl+C to stop the server")
    print("=" * 50)

    try:
        uvicorn.run(
            "back:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
