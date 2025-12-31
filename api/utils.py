#!/usr/bin/env python3
"""
Utilitaires pour l'API SeamlessM4T
"""

import logging
import torch
import numpy as np
import soundfile as sf
import scipy.signal
from pathlib import Path
from typing import Union, Tuple
from fastapi import UploadFile
import tempfile
import os

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_device() -> str:
    """Configure le device (GPU/CPU) automatiquement"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🔥 Utilisation du device: {device}")
    return device


def cleanup_gpu_memory() -> bool:
    """Nettoie la mémoire GPU"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_memory = torch.cuda.mem_get_info()[0] / 1024**3  # en Go
            logger.info(f"🧹 Mémoire GPU nettoyée. Disponible: {free_memory:.1f} Go")
            return True
    except Exception as e:
        logger.warning(f"⚠️ Impossible de nettoyer la mémoire GPU: {e}")
    return False


def save_uploaded_file(upload_file: UploadFile, destination: Path) -> Path:
    """Sauvegarde un fichier uploadé"""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as buffer:
            buffer.write(upload_file.file.read())
        return destination
    except Exception as e:
        logger.error(f"❌ Erreur de sauvegarde du fichier: {e}")
        raise


def load_audio_file(audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """Charge un fichier audio et retourne les données et le sample rate"""
    try:
        audio_data, sample_rate = sf.read(audio_path)
        
        # Conversion stéréo → mono si nécessaire
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Conversion au format float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalisation
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        return audio_data, sample_rate
    except Exception as e:
        logger.error(f"❌ Erreur de chargement audio: {e}")
        raise ValueError(f"Impossible de charger le fichier audio: {e}")


def resample_audio(audio_data: np.ndarray, original_rate: int, target_rate: int = 16000) -> np.ndarray:
    """Resample l'audio vers la fréquence cible"""
    try:
        if original_rate == target_rate:
            return audio_data
        
        original_length = len(audio_data)
        target_length = int(original_length * target_rate / original_rate)
        resampled_audio = scipy.signal.resample(audio_data, target_length)
        logger.info(f"Resampling de {original_rate}Hz à {target_rate}Hz")
        return resampled_audio
    except Exception as e:
        logger.error(f"❌ Erreur de resampling: {e}")
        raise ValueError(f"Erreur de resampling audio: {e}")


def save_audio_file(audio_data: np.ndarray, output_path: Path, sample_rate: int = 16000) -> Path:
    """Sauvegarde un fichier audio"""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Conversion au format compatible
        if audio_data.dtype == np.float16:
            audio_data = audio_data.astype(np.float32)
        
        sf.write(output_path, audio_data, sample_rate)
        return output_path
    except Exception as e:
        logger.error(f"❌ Erreur de sauvegarde audio: {e}")
        raise ValueError(f"Impossible de sauvegarder l'audio: {e}")


def create_temp_audio_file(audio_data: np.ndarray, sample_rate: int = 16000) -> Path:
    """Crée un fichier audio temporaire"""
    temp_dir = Path(tempfile.gettempdir()) / "seamlessm4t"
    temp_dir.mkdir(exist_ok=True)
    
    temp_file = temp_dir / f"temp_audio_{os.urandom(4).hex()}.wav"
    save_audio_file(audio_data, temp_file, sample_rate)
    return temp_file


def validate_audio_duration(audio_data: np.ndarray, sample_rate: int, max_duration: int) -> bool:
    """Valide la durée de l'audio"""
    duration = len(audio_data) / sample_rate
    if duration > max_duration:
        logger.warning(f"⚠️ Audio trop long ({duration:.1f}s > {max_duration}s)")
        return False
    return True


def validate_text_length(text: str, max_length: int) -> bool:
    """Valide la longueur du texte"""
    if len(text) > max_length:
        logger.warning(f"⚠️ Texte trop long ({len(text)} > {max_length} caractères)")
        return False
    return True


def split_audio_into_segments(audio_data: np.ndarray, sample_rate: int, max_duration: int) -> list:
    """Découpe l'audio en segments"""
    segment_length = sample_rate * max_duration
    segments = [audio_data[i:i+segment_length] for i in range(0, len(audio_data), segment_length)]
    logger.info(f"Audio découpé en {len(segments)} segments")
    return segments


def split_text_into_segments(text: str, max_length: int) -> list:
    """Découpe le texte en segments"""
    segments = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    logger.info(f"Texte découpé en {len(segments)} segments")
    return segments


def concatenate_audio_files(audio_files: list, output_path: Path) -> Path:
    """Concatenation de plusieurs fichiers audio"""
    try:
        all_audios = []
        first_sample_rate = None
        
        for audio_file in audio_files:
            audio_data, sample_rate = load_audio_file(audio_file)
            if first_sample_rate is None:
                first_sample_rate = sample_rate
            all_audios.append(audio_data)
            # Nettoyer le fichier temporaire
            os.remove(audio_file)
        
        concatenated_audio = np.concatenate(all_audios)
        save_audio_file(concatenated_audio, output_path, first_sample_rate)
        return output_path
    except Exception as e:
        logger.error(f"❌ Erreur de concatenation: {e}")
        raise ValueError(f"Impossible de concaténer les audios: {e}")


def get_language_name(language_code: str) -> str:
    """Retourne le nom complet d'une langue"""
    from config import SUPPORTED_LANGUAGES
    return SUPPORTED_LANGUAGES.get(language_code, language_code)


def validate_language(language_code: str, supported_languages: dict) -> bool:
    """Valide qu'une langue est supportée"""
    if language_code not in supported_languages:
        supported_codes = ", ".join(supported_languages.keys())
        raise ValueError(f"Langue non supportée: {language_code}. Langues supportées: {supported_codes}")
    return True
