#!/usr/bin/env python3
"""
General configuration for SeamlessM4T API
"""

import os
from pathlib import Path

# Base configuration
APP_NAME = "SeamlessM4T API"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Unified API for SeamlessM4T v2 services"

# Paths
BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "audio_files"
OUTPUT_DIR = BASE_DIR / "output_files"

# Create directories if needed
AUDIO_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_NAME = "facebook/seamless-m4t-v2-large"
SAMPLING_RATE = 16000

# Supported languages
SUPPORTED_LANGUAGES = {
    "arb": "Arabic",
    "ben": "Bengali",
    "cat": "Catalan",
    "ces": "Czech",
    "cmn": "Mandarin Chinese",
    "cym": "Welsh",
    "dan": "Danish",
    "deu": "German",
    "eng": "English",
    "est": "Estonian",
    "fin": "Finnish",
    "fra": "French",
    "hin": "Hindi",
    "ind": "Indonesian",
    "ita": "Italian",
    "jpn": "Japanese",
    "kan": "Kannada",
    "kor": "Korean",
    "mlt": "Maltese",
    "nld": "Dutch",
    "pes": "Western Persian",
    "pol": "Polish",
    "por": "Portuguese",
    "ron": "Romanian",
    "rus": "Russian",
    "slk": "Slovak",
    "spa": "Spanish",
    "swe": "Swedish",
    "swh": "Swahili",
    "tam": "Tamil",
    "tel": "Telugu",
    "tgl": "Tagalog",
    "tha": "Thai",
    "tur": "Turkish",
    "ukr": "Ukrainian",
    "urd": "Urdu",
    "uzn": "Northern Uzbek",
    "vie": "Vietnamese"
}

# API configuration
API_PREFIX = "/api/v1"
MAX_AUDIO_DURATION = 60  # seconds
MAX_TEXT_LENGTH = 5000   # characters

# GPU configuration
USE_GPU = True
GPU_CLEANUP_INTERVAL = 5  # Cleanup after N requests

# FastAPI configuration
FASTAPI_DEBUG = os.getenv("FASTAPI_DEBUG", "False").lower() == "true"
FASTAPI_HOST = "0.0.0.0"
FASTAPI_PORT = 8000

# Gradio configuration
GRADIO_SERVER_NAME = "0.0.0.0"
GRADIO_SERVER_PORT = 7860
GRADIO_TITLE = "SeamlessM4T API Tester"
