#!/usr/bin/env python3
"""
Service T2ST (Text-to-Speech Translation)
"""

import logging
import numpy as np
import torch
from typing import Optional
from pathlib import Path
from ..utils import (
    save_audio_file,
    validate_text_length,
    split_text_into_segments,
    validate_language
)
from .base_service import SeamlessM4TService
from config import SUPPORTED_LANGUAGES, MAX_TEXT_LENGTH

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class T2STService(SeamlessM4TService):
    """Service pour la synthèse vocale Text-to-Speech"""
    
    def __init__(self):
        super().__init__()
        logger.info("✅ Service T2ST initialisé")
    
    def generate_speech(
        self, 
        text: str, 
        src_lang: str, 
        tgt_lang: str,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Génère de la parole à partir de texte
        
        Args:
            text: Texte à synthétiser
            src_lang: Langue source (code ISO 639-3)
            tgt_lang: Langue cible (code ISO 639-3)
            output_path: Chemin de sortie (optionnel)
            
        Returns:
            Chemin vers le fichier audio généré
        """
        try:
            self._increment_request_count()
            
            # Validation des langues
            validate_language(src_lang, SUPPORTED_LANGUAGES)
            validate_language(tgt_lang, SUPPORTED_LANGUAGES)
            
            # Validation du texte
            if not text or not text.strip():
                raise ValueError("Le texte ne peut pas être vide")
            
            # Vérification de la longueur du texte
            if not validate_text_length(text, MAX_TEXT_LENGTH):
                # Découper le texte en segments
                segments = split_text_into_segments(text, MAX_TEXT_LENGTH)
                temp_files = []
                
                for i, segment in enumerate(segments):
                    logger.info(f"Traitement segment {i+1}/{len(segments)}")
                    segment_output = self._generate_single_speech(segment, src_lang, tgt_lang)
                    temp_files.append(segment_output)
                
                # Concatenation des résultats
                if output_path is None:
                    output_path = self._create_temp_file(".wav")
                
                from ..utils import concatenate_audio_files
                return concatenate_audio_files(temp_files, output_path)
            else:
                # Traitement normal pour les textes courts
                return self._generate_single_speech(text, src_lang, tgt_lang, output_path)
            
        except Exception as e:
            logger.error(f"❌ Erreur T2ST: {e}")
            raise
    
    def _generate_single_speech(
        self, 
        text: str, 
        src_lang: str, 
        tgt_lang: str,
        output_path: Optional[Path] = None
    ) -> Path:
        """Génère de la parole pour un seul texte"""
        try:
            # Préparer les entrées pour le modèle
            inputs = self.processor(
                text=[text],
                src_lang=src_lang,
                return_tensors="pt"
            ).to(self.device)
            
            # Générer l'audio
            with torch.no_grad():
                output = self.model.generate(**inputs, tgt_lang=tgt_lang)
            
            # Extraire l'audio de sortie
            if isinstance(output, tuple):
                audio_values = output[0].cpu().numpy().squeeze()
            else:
                audio_values = output["audio_values"][0].cpu().numpy().squeeze()
            
            # Conversion au format compatible
            if audio_values.dtype == np.float16:
                audio_values = audio_values.astype(np.float32)
            
            # Sauvegarde
            if output_path is None:
                output_path = self._create_temp_file(".wav")
            
            save_audio_file(audio_values, output_path, 16000)
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Erreur de génération audio: {e}")
            raise
    
    def get_supported_languages(self) -> dict:
        """Retourne les langues supportées pour T2ST"""
        return SUPPORTED_LANGUAGES
