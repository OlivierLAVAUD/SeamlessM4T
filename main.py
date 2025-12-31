#!/usr/bin/env python3
"""
Main entry point for SeamlessM4T API
"""

import argparse
import logging
import threading
import time
import os
import sys

# Ajouter le chemin courant au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.main import run_api
from app.gradio_app import run_gradio_app
from config import FASTAPI_PORT, GRADIO_SERVER_PORT

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_api_in_thread():
    """Run the API in a separate thread"""
    try:
        logger.info("🚀 Launching FastAPI in a separate thread...")
        run_api()
    except Exception as e:
        logger.error(f"❌ Error in API thread: {e}")


def run_gradio_in_thread():
    """Run the Gradio application in a separate thread"""
    try:
        logger.info("🚀 Launching Gradio application in a separate thread...")
        run_gradio_app()
    except Exception as e:
        logger.error(f"❌ Error in Gradio thread: {e}")


def wait_for_api(timeout: int = 30) -> bool:
    """Wait for the API to be available"""
    import requests
    from config import FASTAPI_HOST, FASTAPI_PORT
    
    api_url = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/api/v1/health"
    
    for _ in range(timeout):
        try:
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                logger.info("✅ FastAPI available")
                return True
        except:
            time.sleep(1)
    
    logger.error("❌ FastAPI not available after waiting")
    return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="SeamlessM4T API - Unified interface for translation services"
    )
    
    parser.add_argument(
        "--api",
        action="store_true",
        help="Run only FastAPI"
    )
    
    parser.add_argument(
        "--gradio",
        action="store_true",
        help="Run only Gradio application"
    )
    
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run both API and Gradio application"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.info("🐛 Debug mode enabled")
    
    logger.info("🚀 Starting SeamlessM4T API")
    logger.info("=" * 50)
    
    if args.both or (not args.api and not args.gradio and not args.both):
        # Run both by default
        logger.info("🎯 Mode: API + Gradio")
        
        # Run API in a thread
        api_thread = threading.Thread(target=run_api_in_thread, daemon=True)
        api_thread.start()
        
        # Wait for API to be available
        if wait_for_api():
            # Run Gradio in main thread
            run_gradio_app()
        else:
            logger.error("❌ Cannot launch Gradio - API not available")
    
    elif args.api:
        logger.info("🎯 Mode: API only")
        run_api()
    
    elif args.gradio:
        logger.info("🎯 Mode: Gradio only")
        
        # Check if API is available
        if wait_for_api(5):  # Wait max 5 seconds
            run_gradio_app()
        else:
            logger.error("❌ Cannot launch Gradio - API not available")
            logger.info("💡 Please run the API first with: python main.py --api")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
