#!/usr/bin/env python3
"""
Service S2TT (Speech-to-Text Translation)
"""

import logging
import numpy as np
import torch
from typing import Optional
from pathlib import Path
from ..utils import (
    load_audio_file, 
    resample_audio,
    validate_audio_duration,
    split_audio_into_segments,
    validate_language
)
from .base_service import SeamlessM4TService
from config import SUPPORTED_LANGUAGES, MAX_AUDIO_DURATION

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S2TTService(SeamlessM4TService):
    """Service pour la transcription et traduction Speech-to-Text"""
    
    def __init__(self):
        super().__init__()
        logger.info("✅ Service S2TT initialisé")
    
    def transcribe_speech(
        self, 
        audio_path: Path, 
        src_lang: str, 
        tgt_lang: str
    ) -> str:
        """
        Transcrit et traduit un fichier audio en texte
        
        Args:
            audio_path: Chemin vers le fichier audio source
            src_lang: Langue source (code ISO 639-3)
            tgt_lang: Langue cible (code ISO 639-3)
            
        Returns:
            Texte transcrit et traduit
        """
        try:
            self._increment_request_count()
            
            # Validation des langues
            validate_language(src_lang, SUPPORTED_LANGUAGES)
            validate_language(tgt_lang, SUPPORTED_LANGUAGES)
            
            # Charger l'audio
            audio_data, sample_rate = load_audio_file(audio_path)
            
            # Resampling si nécessaire
            if sample_rate != 16000:
                audio_data = resample_audio(audio_data, sample_rate, 16000)
                sample_rate = 16000
            
            # Vérification de la durée
            if not validate_audio_duration(audio_data, sample_rate, MAX_AUDIO_DURATION):
                # Découper l'audio en segments
                segments = split_audio_into_segments(audio_data, sample_rate, MAX_AUDIO_DURATION)
                text_segments = []
                
                for i, segment in enumerate(segments):
                    logger.info(f"Traitement segment {i+1}/{len(segments)}")
                    temp_file = self._create_temp_file(".wav")
                    from ..utils import save_audio_file
                    save_audio_file(segment, temp_file, sample_rate)
                    segment_text = self._transcribe_single_audio(temp_file, src_lang, tgt_lang)
                    text_segments.append(segment_text)
                    # Nettoyer le fichier temporaire
                    temp_file.unlink(missing_ok=True)
                
                return " ".join(text_segments)
            else:
                # Traitement normal pour les audios courts
                return self._transcribe_single_audio(audio_path, src_lang, tgt_lang)
            
        except Exception as e:
            logger.error(f"❌ Erreur S2TT: {e}")
            raise
    
    def _transcribe_single_audio(
        self, 
        audio_path: Path, 
        src_lang: str, 
        tgt_lang: str
    ) -> str:
        """Transcrit un seul fichier audio"""
        try:
            # Charger et préparer l'audio
            audio_data, sample_rate = load_audio_file(audio_path)
            
            # Resampling si nécessaire
            if sample_rate != 16000:
                audio_data = resample_audio(audio_data, sample_rate, 16000)
            
            # Préparer les entrées pour le modèle
            inputs = self.processor(
                audios=[audio_data],
                src_lang=src_lang,
                return_tensors="pt"
            ).to(self.device)
            
            # Générer la transcription
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
            logger.error(f"❌ Erreur de transcription audio: {e}")
            raise
    
    def get_supported_languages(self) -> dict:
        """Retourne les langues supportées pour S2TT"""
        return SUPPORTED_LANGUAGES
