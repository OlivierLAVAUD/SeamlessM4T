#!/usr/bin/env python3
"""
Gestionnaire de modèle partagé pour SeamlessM4T
"""

import logging
from typing import Optional
from transformers import AutoProcessor, SeamlessM4Tv2Model, SeamlessM4TFeatureExtractor
import torch

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeamlessM4TModelManager:
    """Singleton pour gérer le modèle SeamlessM4T partagé"""

    _instance = None
    _model = None
    _processor = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialisation du modèle"""
        try:
            logger.info("🔥 Initialisation du gestionnaire de modèle SeamlessM4T")

            # Configurer le device
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🔥 Utilisation du device: {self._device}")

            # Configurer le feature extractor
            feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
                "facebook/seamless-m4t-v2-large",
                sampling_rate=16000
            )

            # Créer le processeur
            self._processor = AutoProcessor.from_pretrained(
                "facebook/seamless-m4t-v2-large",
                feature_extractor=feature_extractor,
                use_fast=False
            )
            logger.info("✅ Processeur chargé")

            # Charger le modèle
            model_kwargs = {
                "torch_dtype": torch.float16 if self._device == "cuda" else torch.float32,
                "low_cpu_mem_usage": True
            }

            self._model = SeamlessM4Tv2Model.from_pretrained(
                "facebook/seamless-m4t-v2-large",
                **model_kwargs
            ).to(self._device)

            logger.info("✅ Modèle chargé avec succès")

        except Exception as e:
            logger.error(f"❌ Erreur d'initialisation du modèle: {e}")
            raise

    @property
    def model(self):
        """Retourne le modèle partagé"""
        return self._model

    @property
    def processor(self):
        """Retourne le processeur partagé"""
        return self._processor

    @property
    def device(self):
        """Retourne le device utilisé"""
        return self._device

    def cleanup_gpu_memory(self):
        """Nettoie la mémoire GPU"""
        try:
            if self._device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                free_memory = torch.cuda.mem_get_info()[0] / 1024**3  # en Go
                logger.info(f"🧹 Mémoire GPU nettoyée. Disponible: {free_memory:.1f} Go")
                return True
        except Exception as e:
            logger.warning(f"⚠️ Impossible de nettoyer la mémoire GPU: {e}")
        return False

# Initialisation du singleton
model_manager = SeamlessM4TModelManager()