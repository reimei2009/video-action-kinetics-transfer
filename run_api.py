"""
Run API Server - Start FastAPI + Gradio UI

Usage:
    python run_api.py
    
Hoặc với custom host/port:
    python run_api.py --host 0.0.0.0 --port 8080
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """Main entry point"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Video Action Recognition API Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development mode)"
    )
    parser.add_argument(
        "--no-gradio",
        action="store_true",
        help="Disable Gradio UI (API only)"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "=" * 70)
    print("🎥 Video Action Recognition API Server")
    print("=" * 70)
    print(f"FastAPI: http://{args.host}:{args.port}")
    print(f"API Docs: http://{args.host}:{args.port}/docs")
    
    if not args.no_gradio:
        print(f"Gradio UI: http://{args.host}:{args.port}/demo")
    
    print("=" * 70 + "\n")
    
    # Import FastAPI app
    from app.main import app
    
    # Mount Gradio UI (nếu không disable)
    if not args.no_gradio:
        try:
            from app.frontend.gradio_app import mount_gradio_app
            app = mount_gradio_app(app)
        except Exception as e:
            print(f"⚠️ Warning: Could not mount Gradio UI: {e}")
            print("   API will run without Gradio interface")
    
    # Start uvicorn server
    import uvicorn
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
