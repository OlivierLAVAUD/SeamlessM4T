#!/usr/bin/env python3
"""
Service de base pour SeamlessM4T
"""

import logging
import torch
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile
import os
from .model_manager import model_manager  # Import du singleton

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeamlessM4TService:
    """Service de base pour les fonctionnalités SeamlessM4T"""

    def __init__(self):
        """Initialisation du service"""
        self.request_count = 0
        logger.info("✅ Service SeamlessM4T initialisé")

    @property
    def device(self):
        """Retourne le device utilisé"""
        return model_manager.device

    @property
    def processor(self):
        """Retourne le processeur partagé"""
        return model_manager.processor

    @property
    def model(self):
        """Retourne le modèle partagé"""
        return model_manager.model

    def _increment_request_count(self) -> None:
        """Incrémente le compteur de requêtes et nettoie si nécessaire"""
        self.request_count += 1
        if self.device == "cuda" and self.request_count % 5 == 0:
            model_manager.cleanup_gpu_memory()

    def get_health_status(self) -> Dict[str, Any]:
        """Retourne l'état de santé du service"""
        status = {
            "model_loaded": model_manager.model is not None,
            "device": self.device,
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory": None
        }

        if torch.cuda.is_available():
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                free_memory = torch.cuda.mem_get_info()[0] / 1024**3
                status["gpu_memory"] = {
                    "total": round(total_memory, 2),
                    "free": round(free_memory, 2),
                    "used": round(total_memory - free_memory, 2)
                }
            except:
                pass

        return status

    def _create_temp_file(self, suffix: str = ".wav") -> Path:
        """Crée un fichier temporaire"""
        temp_dir = Path(tempfile.gettempdir()) / "seamlessm4t"
        temp_dir.mkdir(exist_ok=True)
        return temp_dir / f"temp_{os.urandom(4).hex()}{suffix}"

    def close(self) -> None:
        """Fermeture du service"""
        try:
            model_manager.cleanup_gpu_memory()
            logger.info("🧹 Service SeamlessM4T fermé")
        except Exception as e:
            logger.error(f"❌ Erreur de fermeture: {e}")
