#!/usr/bin/env python3
"""
Service T2TT (Text-to-Text Translation)
"""

import logging
import torch
from typing import Optional
from ..utils import (
    validate_text_length,
    split_text_into_segments,
    validate_language
)
from .base_service import SeamlessM4TService
from config import SUPPORTED_LANGUAGES, MAX_TEXT_LENGTH

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class T2TTService(SeamlessM4TService):
    """Service pour la traduction textuelle Text-to-Text"""
    
    def __init__(self):
        super().__init__()
        logger.info("✅ Service T2TT initialisé")
    
    def translate_text(
        self, 
        text: str, 
        src_lang: str, 
        tgt_lang: str
    ) -> str:
        """
        Traduit un texte d'une langue à une autre
        
        Args:
            text: Texte à traduire
            src_lang: Langue source (code ISO 639-3)
            tgt_lang: Langue cible (code ISO 639-3)
            
        Returns:
            Texte traduit
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
                translated_segments = []
                
                for i, segment in enumerate(segments):
                    logger.info(f"Traitement segment {i+1}/{len(segments)}")
                    try:
                        segment_translation = self._translate_single_text(segment, src_lang, tgt_lang)
                        translated_segments.append(segment_translation)
                    except Exception as segment_error:
                        logger.error(f"❌ Erreur de traduction du segment {i+1}: {segment_error}")
                        translated_segments.append(f"[ERREUR: {str(segment_error)}]")
                
                return " ".join(translated_segments)
            else:
                # Traitement normal pour les textes courts
                return self._translate_single_text(text, src_lang, tgt_lang)
            
        except Exception as e:
            logger.error(f"❌ Erreur T2TT: {e}")
            raise
    
    def _translate_single_text(
        self, 
        text: str, 
        src_lang: str, 
        tgt_lang: str
    ) -> str:
        """Traduit un seul texte"""
        try:
            # Préparer les entrées pour le modèle
            inputs = self.processor(
                text=[text],
                src_lang=src_lang,
                return_tensors="pt"
            ).to(self.device)
            
            # Générer la traduction
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    tgt_lang=tgt_lang,
                    generate_speech=False  # Désactiver la génération audio
                )
            
            # Extraire le texte généré
            if hasattr(output, 'sequences'):
                generated_tokens = output.sequences
            elif isinstance(output, tuple):
                generated_tokens = output[0]
            elif hasattr(output, 'cpu'):
                generated_tokens = output.cpu()
            else:
                generated_tokens = output
            
            # Convertir en texte
            text = self.processor.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )
            
            if isinstance(text, list) and len(text) > 0:
                return text[0]
            return ""
            
        except Exception as e:
            logger.error(f"❌ Erreur de traduction textuelle: {e}")
            raise
    
    def get_supported_languages(self) -> dict:
        """Retourne les langues supportées pour T2TT"""
        return SUPPORTED_LANGUAGES
