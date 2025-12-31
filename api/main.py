#!/usr/bin/env python3
"""
API FastAPI pour SeamlessM4T
"""

import logging
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
import uvicorn

# Import des modèles et services
from .models import (
    S2STRequest, S2TTRequest, T2STRequest, T2TTRequest,
    TranslationResponse, AudioTranslationResponse, TextTranslationResponse,
    HealthCheckResponse, SupportedLanguagesResponse, LanguageInfo, ErrorResponse
)
from .services import S2STService, S2TTService, T2STService, T2TTService
from .services.model_manager import model_manager  # Import du singleton
from config import (
    APP_NAME, APP_VERSION, APP_DESCRIPTION, 
    API_PREFIX, FASTAPI_HOST, FASTAPI_PORT, FASTAPI_DEBUG,
    SUPPORTED_LANGUAGES, AUDIO_DIR, OUTPUT_DIR
)
from .utils import save_uploaded_file, validate_language

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation du modèle partagé (se fait automatiquement via le singleton)
# Les services utiliseront automatiquement le modèle partagé
s2st_service = S2STService()
s2tt_service = S2TTService()
t2st_service = T2STService()
t2tt_service = T2TTService()

# Création de l'application FastAPI
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    debug=FASTAPI_DEBUG
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montage des répertoires statiques
app.mount("/audio_files", StaticFiles(directory=AUDIO_DIR), name="audio_files")
app.mount("/output_files", StaticFiles(directory=OUTPUT_DIR), name="output_files")


@app.get("/", tags=["General"])
async def root():
    """Endpoint racine"""
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "description": APP_DESCRIPTION,
        "api_prefix": API_PREFIX,
        "status": "running"
    }


@app.get(f"{API_PREFIX}/health", tags=["Health"], response_model=HealthCheckResponse)
async def health_check():
    """Vérifie l'état de santé de l'API"""
    try:
        # Vérifier l'état de chaque service
        services_status = {
            "s2st": s2st_service.get_health_status(),
            "s2tt": s2tt_service.get_health_status(),
            "t2st": t2st_service.get_health_status(),
            "t2tt": t2tt_service.get_health_status()
        }
        
        # Déterminer l'état global
        all_healthy = all(
            status["model_loaded"] for status in services_status.values()
        )
        
        status = "healthy" if all_healthy else "degraded"
        
        return {
            "status": status,
            "model_loaded": all_healthy,
            "device": services_status["s2st"]["device"],
            "gpu_available": services_status["s2st"]["gpu_available"],
            "gpu_memory": services_status["s2st"]["gpu_memory"],
            "services": services_status
        }
    except Exception as e:
        logger.error(f"❌ Erreur health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(f"{API_PREFIX}/languages", tags=["General"], response_model=SupportedLanguagesResponse)
async def get_supported_languages():
    """Retourne la liste des langues supportées"""
    try:
        languages = [
            LanguageInfo(code=code, name=name)
            for code, name in SUPPORTED_LANGUAGES.items()
        ]
        return {
            "languages": languages,
            "count": len(languages)
        }
    except Exception as e:
        logger.error(f"❌ Erreur récupération langues: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{API_PREFIX}/s2st", tags=["Speech-to-Speech"], response_model=AudioTranslationResponse)
async def speech_to_speech(
    audio_file: UploadFile = File(...),
    src_lang: str = Form(...),
    tgt_lang: str = Form(...),
    audio_format: Optional[str] = Form("wav")
):
    """Traduit un fichier audio d'une langue à une autre (Speech-to-Speech)"""
    try:
        # Sauvegarder le fichier audio
        audio_path = AUDIO_DIR / f"upload_{os.urandom(4).hex()}.wav"
        save_uploaded_file(audio_file, audio_path)
        
        # Valider les langues
        validate_language(src_lang, SUPPORTED_LANGUAGES)
        validate_language(tgt_lang, SUPPORTED_LANGUAGES)
        
        # Traduire l'audio
        output_path = OUTPUT_DIR / f"s2st_{os.urandom(4).hex()}.wav"
        result_path = s2st_service.translate_speech(
            audio_path=audio_path,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            output_path=output_path
        )
        
        # Nettoyer le fichier temporaire
        audio_path.unlink(missing_ok=True)
        
        # Retourner le résultat
        return {
            "success": True,
            "message": "Traduction audio réussie",
            "result": str(result_path),
            "audio_url": f"/output_files/{result_path.name}",
            "duration": None  # TODO: Calculer la durée
        }
    except Exception as e:
        logger.error(f"❌ Erreur S2ST: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Erreur de traduction audio",
                "error": str(e),
                "audio_url": None
            }
        )


@app.post(f"{API_PREFIX}/s2tt", tags=["Speech-to-Text"], response_model=TextTranslationResponse)
async def speech_to_text(
    audio_file: UploadFile = File(...),
    src_lang: str = Form(...),
    tgt_lang: str = Form(...)
):
    """Transcrit et traduit un fichier audio en texte (Speech-to-Text)"""
    try:
        # Sauvegarder le fichier audio
        audio_path = AUDIO_DIR / f"upload_{os.urandom(4).hex()}.wav"
        save_uploaded_file(audio_file, audio_path)
        
        # Valider les langues
        validate_language(src_lang, SUPPORTED_LANGUAGES)
        validate_language(tgt_lang, SUPPORTED_LANGUAGES)
        
        # Transcrire l'audio
        text_result = s2tt_service.transcribe_speech(
            audio_path=audio_path,
            src_lang=src_lang,
            tgt_lang=tgt_lang
        )
        
        # Nettoyer le fichier temporaire
        audio_path.unlink(missing_ok=True)
        
        # Retourner le résultat
        return {
            "success": True,
            "message": "Transcription réussie",
            "result": text_result,
            "text": text_result,
            "character_count": len(text_result)
        }
    except Exception as e:
        logger.error(f"❌ Erreur S2TT: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Erreur de transcription",
                "error": str(e),
                "text": None
            }
        )


@app.post(f"{API_PREFIX}/t2st", tags=["Text-to-Speech"], response_model=AudioTranslationResponse)
async def text_to_speech(
    request_data: T2STRequest
):
    """Génère de la parole à partir de texte (Text-to-Speech)"""
    try:
        # Valider la requête
        if not request_data.text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Générer l'audio
        output_path = OUTPUT_DIR / f"t2st_{os.urandom(4).hex()}.wav"
        result_path = t2st_service.generate_speech(
            text=request_data.text,
            src_lang=request_data.src_lang,
            tgt_lang=request_data.tgt_lang,
            output_path=output_path
        )
        
        # Retourner le résultat
        return {
            "success": True,
            "message": "Génération audio réussie",
            "result": str(result_path),
            "audio_url": f"/output_files/{result_path.name}",
            "duration": None  # TODO: Calculer la durée
        }
    except Exception as e:
        logger.error(f"❌ Erreur T2ST: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Erreur de génération audio",
                "error": str(e),
                "audio_url": None
            }
        )


@app.post(f"{API_PREFIX}/t2tt", tags=["Text-to-Text"], response_model=TextTranslationResponse)
async def text_to_text(
    request_data: T2TTRequest
):
    """Traduit un texte d'une langue à une autre (Text-to-Text)"""
    try:
        # Valider la requête
        if not request_data.text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Traduire le texte
        text_result = t2tt_service.translate_text(
            text=request_data.text,
            src_lang=request_data.src_lang,
            tgt_lang=request_data.tgt_lang
        )
        
        # Retourner le résultat
        return {
            "success": True,
            "message": "Traduction textuelle réussie",
            "result": text_result,
            "text": text_result,
            "character_count": len(text_result)
        }
    except Exception as e:
        logger.error(f"❌ Erreur T2TT: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Erreur de traduction textuelle",
                "error": str(e),
                "text": None
            }
        )


# Gestion des erreurs globales
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Gestionnaire d'exceptions HTTP"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


def run_api():
    """Lance l'API FastAPI"""
    try:
        logger.info(f"🚀 Lancement de l'API {APP_NAME} v{APP_VERSION}")
        logger.info(f"📍 URL: http://{FASTAPI_HOST}:{FASTAPI_PORT}")
        logger.info(f"📖 Documentation: http://{FASTAPI_HOST}:{FASTAPI_PORT}/docs")
        
        uvicorn.run(
            "api.main:app",
            host=FASTAPI_HOST,
            port=FASTAPI_PORT,
            reload=FASTAPI_DEBUG,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        raise


if __name__ == "__main__":
    run_api()
