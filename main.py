#!/usr/bin/env python3
"""
Point d'entrée principal pour l'API SeamlessM4T
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
    """Lance l'API dans un thread séparé"""
    try:
        logger.info("🚀 Lancement de l'API FastAPI dans un thread séparé...")
        run_api()
    except Exception as e:
        logger.error(f"❌ Erreur dans le thread API: {e}")


def run_gradio_in_thread():
    """Lance l'application Gradio dans un thread séparé"""
    try:
        logger.info("🚀 Lancement de l'application Gradio dans un thread séparé...")
        run_gradio_app()
    except Exception as e:
        logger.error(f"❌ Erreur dans le thread Gradio: {e}")


def wait_for_api(timeout: int = 30) -> bool:
    """Attend que l'API soit disponible"""
    import requests
    from config import FASTAPI_HOST, FASTAPI_PORT
    
    api_url = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/api/v1/health"
    
    for _ in range(timeout):
        try:
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                logger.info("✅ API FastAPI disponible")
                return True
        except:
            time.sleep(1)
    
    logger.error("❌ API FastAPI non disponible après attente")
    return False


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="API SeamlessM4T - Interface unifiée pour les services de traduction"
    )
    
    parser.add_argument(
        "--api",
        action="store_true",
        help="Lancer uniquement l'API FastAPI"
    )
    
    parser.add_argument(
        "--gradio",
        action="store_true",
        help="Lancer uniquement l'application Gradio"
    )
    
    parser.add_argument(
        "--both",
        action="store_true",
        help="Lancer à la fois l'API et l'application Gradio"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activer le mode debug"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.info("🐛 Mode debug activé")
    
    logger.info("🚀 Démarrage de l'API SeamlessM4T")
    logger.info("=" * 50)
    
    if args.both or (not args.api and not args.gradio and not args.both):
        # Lancer les deux par défaut
        logger.info("🎯 Mode: API + Gradio")
        
        # Lancer l'API dans un thread
        api_thread = threading.Thread(target=run_api_in_thread, daemon=True)
        api_thread.start()
        
        # Attendre que l'API soit disponible
        if wait_for_api():
            # Lancer Gradio dans le thread principal
            run_gradio_app()
        else:
            logger.error("❌ Impossible de lancer Gradio - API non disponible")
    
    elif args.api:
        logger.info("🎯 Mode: API uniquement")
        run_api()
    
    elif args.gradio:
        logger.info("🎯 Mode: Gradio uniquement")
        
        # Vérifier si l'API est disponible
        if wait_for_api(5):  # Attendre 5 secondes max
            run_gradio_app()
        else:
            logger.error("❌ Impossible de lancer Gradio - API non disponible")
            logger.info("💡 Veuillez lancer l'API d'abord avec: python main.py --api")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
