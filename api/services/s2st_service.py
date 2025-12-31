#!/usr/bin/env python3
"""
Service S2ST (Speech-to-Speech Translation)
"""

import logging
import numpy as np
import torch
from typing import Optional, Tuple
from pathlib import Path
from ..utils import (
    load_audio_file, 
    resample_audio, 
    save_audio_file,
    validate_audio_duration,
    split_audio_into_segments,
    concatenate_audio_files,
    validate_language
)
from .base_service import SeamlessM4TService
from config import SUPPORTED_LANGUAGES, MAX_AUDIO_DURATION

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S2STService(SeamlessM4TService):
    """Service for Speech-to-Speech translation"""
    
    def __init__(self):
        super().__init__()
        logger.info("✅ Service S2ST initialisé")
    
    def translate_speech(
        self, 
        audio_path: Path, 
        src_lang: str, 
        tgt_lang: str,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Translate an audio file from one language to another
        
        Args:
            audio_path: Path to the source audio file
            src_lang: Source language (ISO 639-3 code)
            tgt_lang: Target language (ISO 639-3 code)
            output_path: Output path (optional)
            
        Returns:
            Chemin vers le fichier audio traduit
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
                temp_files = []
                
                for i, segment in enumerate(segments):
                    logger.info(f"Traitement segment {i+1}/{len(segments)}")
                    temp_file = self._create_temp_file(".wav")
                    save_audio_file(segment, temp_file, sample_rate)
                    segment_output = self._translate_single_audio(temp_file, src_lang, tgt_lang)
                    temp_files.append(segment_output)
                
                # Concatenation des résultats
                if output_path is None:
                    output_path = self._create_temp_file(".wav")
                
                return concatenate_audio_files(temp_files, output_path)
            else:
                # Traitement normal pour les audios courts
                return self._translate_single_audio(audio_path, src_lang, tgt_lang, output_path)
            
        except Exception as e:
            logger.error(f"❌ Erreur S2ST: {e}")
            raise
    
    def _translate_single_audio(
        self, 
        audio_path: Path, 
        src_lang: str, 
        tgt_lang: str,
        output_path: Optional[Path] = None
    ) -> Path:
        """Translate a single audio file"""
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
            
            # Generate the translation
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
            logger.error(f"❌ Audio translation error: {e}")
            raise
    
    def get_supported_languages(self) -> dict:
        """Retourne les langues supportées pour S2ST"""
        return SUPPORTED_LANGUAGES
