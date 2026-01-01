#!/usr/bin/env python3
"""
Gradio application to test SeamlessM4T API
"""

import gradio as gr
import requests
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple
from config import (
    GRADIO_SERVER_NAME, GRADIO_SERVER_PORT, GRADIO_TITLE,
    SUPPORTED_LANGUAGES, FASTAPI_HOST, FASTAPI_PORT
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de l'API
API_BASE_URL = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}"
API_PREFIX = "/api/v1"

# Chemins pour les fichiers temporaires
TEMP_DIR = Path("gradio_temp")
TEMP_DIR.mkdir(exist_ok=True)


def get_api_url(endpoint: str) -> str:
    """Retourne l'URL complète de l'API"""
    return f"{API_BASE_URL}{API_PREFIX}/{endpoint}"


def check_api_health() -> bool:
    """Vérifie que l'API est disponible"""
    try:
        response = requests.get(get_api_url("health"), timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"❌ Impossible de joindre l'API: {e}")
        return False


def get_supported_languages() -> dict:
    """Récupère les langues supportées depuis l'API"""
    try:
        response = requests.get(get_api_url("languages"), timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {lang["code"]: lang["name"] for lang in data["languages"]}
        return SUPPORTED_LANGUAGES
    except Exception as e:
        logger.error(f"❌ Erreur récupération langues: {e}")
        return SUPPORTED_LANGUAGES


def call_api_endpoint(
    endpoint: str,
    method: str = "POST",
    files: Optional[dict] = None,
    data: Optional[dict] = None,
    json: Optional[dict] = None
) -> Tuple[bool, dict]:
    """Appelle un endpoint de l'API"""
    try:
        url = get_api_url(endpoint)
        
        if method.upper() == "GET":
            response = requests.get(url, params=data, timeout=30)
        elif method.upper() == "POST":
            response = requests.post(url, files=files, data=data, json=json, timeout=60)
        else:
            return False, {"error": "Méthode non supportée"}
        
        if response.status_code == 200:
            return True, response.json()
        else:
            error_msg = response.json().get("error", "Erreur inconnue")
            return False, {"error": f"Erreur {response.status_code}: {error_msg}"}
    except Exception as e:
        logger.error(f"❌ Erreur appel API: {e}")
        return False, {"error": str(e)}


def s2st_translation(
    audio_file: str,
    src_lang: str,
    tgt_lang: str
) -> Tuple[Optional[str], Optional[str]]:
    """Speech-to-Speech Translation"""
    if not check_api_health():
        return None, "❌ API not available"
    
    try:
        # Préparer les fichiers et données
        files = {"audio_file": open(audio_file, "rb")}
        data = {
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        }
        
        success, result = call_api_endpoint("s2st", "POST", files=files, data=data)
        
        if success and result.get("success"):
            audio_url = result.get("audio_url")
            if audio_url:
                # Télécharger le fichier audio
                audio_response = requests.get(f"{API_BASE_URL}{audio_url}", timeout=30)
                if audio_response.status_code == 200:
                    output_file = TEMP_DIR / f"s2st_output_{int(time.time())}.wav"
                    with open(output_file, "wb") as f:
                        f.write(audio_response.content)
                    return str(output_file), None
            return None, "✅ Translation successful but unable to download audio"
        else:
            error = result.get("error", "Erreur inconnue")
            return None, f"❌ {error}"
    except Exception as e:
        logger.error(f"❌ Erreur S2ST: {e}")
        return None, f"❌ Erreur: {str(e)}"


def s2tt_translation(
    audio_file: str,
    src_lang: str,
    tgt_lang: str
) -> Tuple[Optional[str], Optional[str]]:
    """Transcription Speech-to-Text"""
    if not check_api_health():
        return None, "❌ API not available"
    
    try:
        files = {"audio_file": open(audio_file, "rb")}
        data = {
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        }
        
        success, result = call_api_endpoint("s2tt", "POST", files=files, data=data)
        
        if success and result.get("success"):
            text = result.get("text", "")
            return text, None
        else:
            error = result.get("error", "Erreur inconnue")
            return None, f"❌ {error}"
    except Exception as e:
        logger.error(f"❌ Erreur S2TT: {e}")
        return None, f"❌ Erreur: {str(e)}"


def t2st_translation(
    text: str,
    src_lang: str,
    tgt_lang: str
) -> Tuple[Optional[str], Optional[str]]:
    """Synthèse vocale Text-to-Speech"""
    if not check_api_health():
        return None, "❌ API not available"
    
    try:
        json_data = {
            "text": text,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        }
        
        success, result = call_api_endpoint("t2st", "POST", json=json_data)
        
        if success and result.get("success"):
            audio_url = result.get("audio_url")
            if audio_url:
                # Télécharger le fichier audio
                audio_response = requests.get(f"{API_BASE_URL}{audio_url}", timeout=30)
                if audio_response.status_code == 200:
                    output_file = TEMP_DIR / f"t2st_output_{int(time.time())}.wav"
                    with open(output_file, "wb") as f:
                        f.write(audio_response.content)
                    return str(output_file), None
            return None, "✅ Génération réussie mais impossible de télécharger l'audio"
        else:
            error = result.get("error", "Erreur inconnue")
            return None, f"❌ {error}"
    except Exception as e:
        logger.error(f"❌ Erreur T2ST: {e}")
        return None, f"❌ Erreur: {str(e)}"


def t2tt_translation(
    text: str,
    src_lang: str,
    tgt_lang: str
) -> Tuple[Optional[str], Optional[str]]:
    """Text translation Text-to-Text"""
    if not check_api_health():
        return None, "❌ API not available"
    
    try:
        json_data = {
            "text": text,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        }
        
        success, result = call_api_endpoint("t2tt", "POST", json=json_data)
        
        if success and result.get("success"):
            text = result.get("text", "")
            return text, None
        else:
            error = result.get("error", "Erreur inconnue")
            return None, f"❌ {error}"
    except Exception as e:
        logger.error(f"❌ Erreur T2TT: {e}")
        return None, f"❌ Erreur: {str(e)}"


def get_api_status() -> str:
    """Retourne l'état de l'API"""
    if check_api_health():
        return "🟢 API is available and operational"
    else:
        return "🔴 API not available"


def create_gradio_interface():
    """Create the Gradio interface"""
    
    # Récupérer les langues supportées
    languages = get_supported_languages()
    language_codes = list(languages.keys())
    language_names = {code: f"{code} - {name}" for code, name in languages.items()}
    
    with gr.Blocks(title=GRADIO_TITLE) as app:
        gr.Markdown(f"""
        # 🎤 {GRADIO_TITLE}
        
        **Massively Multilingual & Multimodal Machine Translation with Hugginface**
        
        {get_api_status()}
        
        **Available functionalities:**
        - 🎤 Speech-to-Speech Translation (S2ST)
        - 📝 Speech-to-Text Translation (S2TT)
        - 💬 Text-to-Speech Translation (T2ST)
        - 📄 Text-to-Text Translation (T2TT)
        
        **API URL:** {API_BASE_URL}
        **Documentation:** {API_BASE_URL}/docs
        
        ---
        """)
        
        with gr.Tabs():
            # Onglet S2ST
            with gr.Tab("🎤 Speech-to-Speech (S2ST)"):
                gr.Markdown("### Speech translation (Audio → Audio)")
                
                with gr.Row():
                    s2st_audio = gr.Audio(
                        label="Audio à traduire",
                        type="filepath",
                        sources=["microphone", "upload"]
                    )
                
                with gr.Row():
                    s2st_src_lang = gr.Dropdown(
                        choices=language_codes,
                        value="fra",
                        label="Source Language",
                        info="Sélectionnez la langue de l'audio source"
                    )
                    s2st_tgt_lang = gr.Dropdown(
                        choices=language_codes,
                        value="eng",
                        label="Target Language",
                        info="Select the translation language"
                    )
                
                s2st_btn = gr.Button("Traduire audio", variant="primary")
                s2st_output = gr.Audio(label="Audio traduit", type="filepath")
                s2st_status = gr.Textbox(label="Statut", interactive=False)
                
                s2st_btn.click(
                    fn=s2st_translation,
                    inputs=[s2st_audio, s2st_src_lang, s2st_tgt_lang],
                    outputs=[s2st_output, s2st_status]
                )
            
            # Onglet S2TT
            with gr.Tab("📝 Speech-to-Text (S2TT)"):
                gr.Markdown("### Transcription vocale (Audio → Texte)")
                
                with gr.Row():
                    s2tt_audio = gr.Audio(
                        label="Audio à transcrire",
                        type="filepath",
                        sources=["microphone", "upload"]
                    )
                
                with gr.Row():
                    s2tt_src_lang = gr.Dropdown(
                        choices=language_codes,
                        value="fra",
                        label="Source Language"
                    )
                    s2tt_tgt_lang = gr.Dropdown(
                        choices=language_codes,
                        value="eng",
                        label="Target Language"
                    )
                
                s2tt_btn = gr.Button("Transcrire audio", variant="primary")
                s2tt_output = gr.Textbox(label="Texte transcrit", lines=5)
                s2tt_status = gr.Textbox(label="Statut", interactive=False)
                
                s2tt_btn.click(
                    fn=s2tt_translation,
                    inputs=[s2tt_audio, s2tt_src_lang, s2tt_tgt_lang],
                    outputs=[s2tt_output, s2tt_status]
                )
            
            # Onglet T2ST
            with gr.Tab("💬 Text-to-Speech (T2ST)"):
                gr.Markdown("### Synthèse vocale (Texte → Audio)")
                
                with gr.Row():
                    t2st_text = gr.Textbox(
                        label="Texte à synthétiser",
                        value="Bonjour le monde",
                        lines=3,
                        placeholder="Entrez le texte à synthétiser..."
                    )
                
                with gr.Row():
                    t2st_src_lang = gr.Dropdown(
                        choices=language_codes,
                        value="fra",
                        label="Source Language"
                    )
                    t2st_tgt_lang = gr.Dropdown(
                        choices=language_codes,
                        value="eng",
                        label="Target Language"
                    )
                
                t2st_btn = gr.Button("Générer audio", variant="primary")
                t2st_output = gr.Audio(label="Audio généré", type="filepath")
                t2st_status = gr.Textbox(label="Statut", interactive=False)
                
                t2st_btn.click(
                    fn=t2st_translation,
                    inputs=[t2st_text, t2st_src_lang, t2st_tgt_lang],
                    outputs=[t2st_output, t2st_status]
                )
            
            # Onglet T2TT
            with gr.Tab("📄 Text-to-Text (T2TT)"):
                gr.Markdown("### Text translation (Text → Text)")
                
                with gr.Row():
                    t2tt_text = gr.Textbox(
                        label="Texte à traduire",
                        lines=5,
                        placeholder="Entrez le texte à traduire..."
                    )
                
                with gr.Row():
                    t2tt_src_lang = gr.Dropdown(
                        choices=language_codes,
                        value="fra",
                        label="Source Language"
                    )
                    t2tt_tgt_lang = gr.Dropdown(
                        choices=language_codes,
                        value="eng",
                        label="Target Language"
                    )
                
                t2tt_btn = gr.Button("Traduire texte", variant="primary")
                t2tt_output = gr.Textbox(label="Texte traduit", lines=5)
                t2tt_status = gr.Textbox(label="Statut", interactive=False)
                
                t2tt_btn.click(
                    fn=t2tt_translation,
                    inputs=[t2tt_text, t2tt_src_lang, t2tt_tgt_lang],
                    outputs=[t2tt_output, t2tt_status]
                )
            
            # Onglet Informations
            with gr.Tab("ℹ️ Informations"):
                gr.Markdown(f"""
                ## API Description
                
                **API Status:** {get_api_status()}
                
                **Available Endpoints:**
                - `GET /api/v1/health` - Check Health
                - `GET /api/v1/languages` - Supported Language List
                - `POST /api/v1/s2st` - Speech-to-Speech Translation
                - `POST /api/v1/s2tt` - Speech-to-Text Translation
                - `POST /api/v1/t2st` - Text-to-Speech Translation
                - `POST /api/v1/t2tt` - Text-to-Text Translation
                
                **Supported Languages: ({len(languages)} languages):**
                """)
                
                # Afficher les langues sous forme de tableau
                lang_table = "| Code | Name         |\n|------|-------------|\n"
                for code, name in sorted(languages.items()):
                    lang_table += f"| `{code}` | {name} |\n"
                
                gr.Markdown(lang_table)
                
                gr.Markdown(f"""
                **Configuration:**
                - **API URL:** `{API_BASE_URL}`
                - **Port API:** `{FASTAPI_PORT}`
                - **Port Gradio:** `{GRADIO_SERVER_PORT}`
                
                **Documentation:**
                - [FastAPI Docs]({API_BASE_URL}/docs)
                - [ReDoc]({API_BASE_URL}/redoc)
                
                ---
                
                **Note:**
                This interface allows you to test all features of the SeamlessM4T v2 API.
                Les requêtes sont envoyées à l'API FastAPI qui effectue le traitement réel.
                
                © 2024 SeamlessM4T API
                """)
        
        # Pied de page
        gr.Markdown(f"""
        ---
        **Statut:** {get_api_status()} | **Version:** 1.0.0
        """)
    
    return app


def run_gradio_app():
    """Lance l'application Gradio"""
    try:
        logger.info(f"🚀 Lancement de l'application Gradio")
        logger.info(f"📍 URL: http://{GRADIO_SERVER_NAME}:{GRADIO_SERVER_PORT}")
        
        app = create_gradio_interface()
        app.launch(
            server_name=GRADIO_SERVER_NAME,
            server_port=GRADIO_SERVER_PORT,
            share=False,
            show_error=True
        )
    except Exception as e:
        logger.error(f"❌ Erreur fatale Gradio: {e}")
        raise


if __name__ == "__main__":
    run_gradio_app()
