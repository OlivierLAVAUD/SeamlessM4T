#!/usr/bin/env python3
"""
Seamless TTS v1 - Synthèse vocale spécialisée avec auto-détection GPU
"""

import gradio as gr
import torch
import logging
from transformers import AutoProcessor, SeamlessM4Tv2Model
import numpy as np
import soundfile as sf
import os
from datetime import datetime

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeamlessM4T:
    """Synthèse vocale spécialisée"""
    
    # Langues supportées par le modèle SeamlessM4T v2 pour la synthèse vocale
    TTS_SUPPORTED_LANGUAGES = {
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
    
    # Toutes les langues supportées par SeamlessM4T (pour référence)
    ALL_SUPPORTED_LANGUAGES = {
        "afr": "Afrikaans",
        "amh": "Amharic",
        "ara": "Arabic",
        "arb": "Arabic (Modern Standard)",
        "asm": "Assamese",
        "aze": "Azerbaijani",
        "bel": "Belarusian",
        "ben": "Bengali",
        "bos": "Bosnian",
        "bul": "Bulgarian",
        "cat": "Catalan",
        "ceb": "Cebuano",
        "ces": "Czech",
        "ckb": "Central Kurdish",
        "cmn": "Mandarin Chinese",
        "cym": "Welsh",
        "dan": "Danish",
        "deu": "German",
        "ell": "Greek",
        "eng": "English",
        "est": "Estonian",
        "eus": "Basque",
        "fin": "Finnish",
        "fra": "French",
        "gaz": "West Central Oromo",
        "gle": "Irish",
        "glg": "Galician",
        "guj": "Gujarati",
        "heb": "Hebrew",
        "hin": "Hindi",
        "hrv": "Croatian",
        "hun": "Hungarian",
        "hye": "Armenian",
        "ibo": "Igbo",
        "ind": "Indonesian",
        "isl": "Icelandic",
        "ita": "Italian",
        "jav": "Javanese",
        "jpn": "Japanese",
        "kan": "Kannada",
        "kat": "Georgian",
        "kaz": "Kazakh",
        "kea": "Kabuverdianu",
        "khk": "Halh Mongolian",
        "khm": "Khmer",
        "kir": "Kyrgyz",
        "kor": "Korean",
        "lao": "Lao",
        "lit": "Lithuanian",
        "ltz": "Luxembourgish",
        "lug": "Ganda",
        "luo": "Luo",
        "lvs": "Standard Latvian",
        "mai": "Maithili",
        "mal": "Malayalam",
        "mar": "Marathi",
        "mkd": "Macedonian",
        "mlt": "Maltese",
        "mni": "Meitei",
        "mya": "Burmese",
        "nld": "Dutch",
        "nno": "Norwegian Nynorsk",
        "nob": "Norwegian Bokmål",
        "npi": "Nepali",
        "nya": "Nyanja",
        "oci": "Occitan",
        "ory": "Odia",
        "pan": "Punjabi",
        "pbt": "Southern Pashto",
        "pes": "Western Persian",
        "pol": "Polish",
        "por": "Portuguese",
        "ron": "Romanian",
        "rus": "Russian",
        "slk": "Slovak",
        "slv": "Slovenian",
        "sna": "Shona",
        "snd": "Sindhi",
        "som": "Somali",
        "spa": "Spanish",
        "srp": "Serbian",
        "swe": "Swedish",
        "swh": "Swahili",
        "tam": "Tamil",
        "tel": "Telugu",
        "tgk": "Tajik",
        "tgl": "Tagalog",
        "tha": "Thai",
        "tur": "Turkish",
        "ukr": "Ukrainian",
        "urd": "Urdu",
        "uzb": "Uzbek",
        "uzn": "Northern Uzbek",
        "vie": "Vietnamese",
        "xho": "Xhosa",
        "yid": "Yiddish",
        "yor": "Yoruba",
        "yue": "Cantonese",
        "zho": "Chinese",
        "zsm": "Standard Malay",
        "zul": "Zulu"
    }
    
    def __init__(self):
        """Initialisation avec auto-détection GPU"""
        logger.info("🔋 Initialisation de SeamlessM4T...")
        
        # Auto-détection GPU/CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🔥 Utilisation du device: {self.device}")
        
        # Chargement du modèle
        self._load_model()
    
    def _load_model(self):
        """Chargement optimisé du modèle"""
        try:
            logger.info("🔋 Chargement du modèle SeamlessM4T...")
            
            # Charger le processeur
            self.processor = AutoProcessor.from_pretrained(
                "facebook/seamless-m4t-v2-large",
                use_fast=False
            )
            logger.info("✅ Processeur chargé")
            
            # Charger le modèle avec optimisation
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "low_cpu_mem_usage": True
            }
            
            self.model = SeamlessM4Tv2Model.from_pretrained(
                "facebook/seamless-m4t-v2-large",
                **model_kwargs
            ).to(self.device)
            
            logger.info("✅ Modèle chargé avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur de chargement: {e}")
            raise
    
    def generate_speech(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Génération de parole (TTS) avec gestion des textes longs"""
        try:
            # Validation
            if src_lang not in self.ALL_SUPPORTED_LANGUAGES:
                raise ValueError(f"Langue source non supportée: {src_lang}")
            if tgt_lang not in self.TTS_SUPPORTED_LANGUAGES:
                supported_tts_langs = ", ".join(self.TTS_SUPPORTED_LANGUAGES.keys())
                raise ValueError(f"Langue cible non supportée pour TTS: {tgt_lang}. "
                               f"Langues supportées: {supported_tts_langs}")
            
            # Vérification de la longueur du texte
            max_length = 500  # Limite pour éviter les problèmes mémoire
            if len(text) > max_length:
                logger.warning(f"Texte trop long ({len(text)} caractères), découpage en segments...")
                # Découper le texte en segments
                segments = [text[i:i+max_length] for i in range(0, len(text), max_length)]
                audio_paths = []
                
                for i, segment in enumerate(segments):
                    logger.info(f"Traitement segment {i+1}/{len(segments)}")
                    inputs = self.processor(
                        text=segment,
                        src_lang=src_lang,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        try:
                            output = self.model.generate(**inputs, tgt_lang=tgt_lang)
                        except Exception as model_error:
                            # Gestion des erreurs spécifiques du modèle
                            error_msg = str(model_error)
                            if "not supported by this model" in error_msg:
                                # Extraire la liste des langues supportées de l'erreur
                                start_idx = error_msg.find("in ") + 3
                                end_idx = error_msg.find(". Note that")
                                if start_idx > 0 and end_idx > 0:
                                    supported_langs = error_msg[start_idx:end_idx].strip()
                                    raise ValueError(f"Langue cible '{tgt_lang}' non supportée pour TTS. "
                                                   f"Langues supportées: {supported_langs}")
                            raise
                    
                    audio_values = output[0].cpu().numpy().squeeze()
                    
                    # Conversion au format compatible avec soundfile
                    if audio_values.dtype == np.float16:
                        audio_values = audio_values.astype(np.float32)
                    
                    # Sauvegarde du segment
                    segment_path = self._save_audio(audio_values)
                    audio_paths.append(segment_path)
                
                # Concatenation des segments audio
                all_audios = []
                first_sample_rate = None
                for path in audio_paths:
                    audio_data, sample_rate = sf.read(path)
                    if first_sample_rate is None:
                        first_sample_rate = sample_rate
                    all_audios.append(audio_data)
                
                # Concatenation
                concatenated_audio = np.concatenate(all_audios)
                
                # Sauvegarde du résultat final
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_output_path = f"tts_output/tts_concatenated_{timestamp}.wav"
                sf.write(final_output_path, concatenated_audio, first_sample_rate)
                
                # Nettoyage des segments temporaires
                for path in audio_paths:
                    os.remove(path)
                
                return final_output_path
            else:
                # Traitement normal pour les textes courts
                inputs = self.processor(
                    text=text,
                    src_lang=src_lang,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    try:
                        output = self.model.generate(**inputs, tgt_lang=tgt_lang)
                    except Exception as model_error:
                        # Gestion des erreurs spécifiques du modèle
                        error_msg = str(model_error)
                        if "not supported by this model" in error_msg:
                            # Extraire la liste des langues supportées de l'erreur
                            start_idx = error_msg.find("in ") + 3
                            end_idx = error_msg.find(". Note that")
                            if start_idx > 0 and end_idx > 0:
                                supported_langs = error_msg[start_idx:end_idx].strip()
                                raise ValueError(f"Langue cible '{tgt_lang}' non supportée pour TTS. "
                                               f"Langues supportées: {supported_langs}")
                        raise
                
                audio_values = output[0].cpu().numpy().squeeze()
                
                # Conversion au format compatible avec soundfile
                if audio_values.dtype == np.float16:
                    audio_values = audio_values.astype(np.float32)
                
                # Sauvegarde
                output_path = self._save_audio(audio_values)
                
                return output_path
            
        except Exception as e:
            logger.error(f"❌ Erreur de génération: {e}")
            raise
    
    def _save_audio(self, audio: np.ndarray) -> str:
        """Sauvegarde de l'audio généré"""
        os.makedirs("tts_output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"tts_output/tts_{timestamp}.wav"
        sf.write(output_path, audio, 16000)
        return output_path


class SeamlessTTSApp:
    """Interface Gradio pour SeamlessM4T"""
    
    def __init__(self):
        self.tts = SeamlessM4T()
        # Utiliser uniquement les langues supportées pour TTS
        self.languages = self.tts.TTS_SUPPORTED_LANGUAGES
    
    def tts_interface(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Interface pour la synthèse vocale"""
        try:
            output_path = self.tts.generate_speech(text, src_lang, tgt_lang)
            return output_path
        except ValueError as e:
            # Erreur de validation des langues - limiter la longueur du message
            error_msg = str(e)
            if len(error_msg) > 200:
                # Extraire les informations essentielles
                if "Langues supportées:" in error_msg:
                    start_idx = error_msg.find("Langues supportées:")
                    return error_msg[:start_idx + 20] + "... (voir logs pour la liste complète)"
                else:
                    return error_msg[:200] + "..."
            return f"❌ {error_msg}"
        except Exception as e:
            # Autres erreurs - limiter la longueur
            error_msg = str(e)
            if len(error_msg) > 200:
                return f"❌ {error_msg[:200]}..."
            return f"❌ Erreur inattendue: {error_msg}"
    
    def create_interface(self):
        """Crée l'interface Gradio"""
        
        with gr.Blocks(title="Seamless TTS v1") as app:
            gr.Markdown("""
            # 🎤 SeamlessM4T  Text-to-speech translation (T2ST)        
            Synthèse vocale spécialisée avec auto-détection GPU
            
            **Fonctionnalités:**
            - 🎤 Texte vers Parole (TTS)
            - 🔥 Auto-détection GPU/CPU
            - 🌍 Support multilingue (36 langues)
            - 💾 Sauvegarde automatique
            
            **Langues supportées pour TTS:** Arabic, Bengali, Catalan, Czech, Mandarin, Welsh, Danish, German, English, Estonian, Finnish, French, Hindi, Indonesian, Italian, Japanese, Kannada, Korean, Maltese, Dutch, Persian, Polish, Portuguese, Romanian, Russian, Slovak, Spanish, Swedish, Swahili, Tamil, Telugu, Tagalog, Thai, Turkish, Ukrainian, Urdu, Uzbek, Vietnamese
            """)
            
            with gr.Row():
                tts_text = gr.Textbox(
                    label="Texte à synthétiser",
                    value="Bonjour le monde",
                    lines=3
                )
            
            with gr.Row():
                tts_src_lang = gr.Dropdown(
                    choices=list(self.languages.keys()),
                    value="fra",
                    label="Langue source"
                )
                tts_tgt_lang = gr.Dropdown(
                    choices=list(self.languages.keys()),
                    value="eng",
                    label="Langue cible"
                )
            
            tts_btn = gr.Button("Générer audio", variant="primary")
            tts_output = gr.Audio(label="Audio généré", type="filepath")
            
            tts_btn.click(
                fn=self.tts_interface,
                inputs=[tts_text, tts_src_lang, tts_tgt_lang],
                outputs=tts_output
            )
            
            gr.Markdown("""
            ---
            ### Informations
            - **Device:** """ + ("🔥 GPU" if torch.cuda.is_available() else "❄️ CPU") + """
            - **Modèle:** facebook/seamless-m4t-v2-large
            - **Échantillonnage:** 16kHz
            - **Format:** WAV
            
            © 2024 SeamlessM4T            """)
        
        return app
    
    def launch(self):
        """Lance l'application"""
        app = self.create_interface()
        app.launch(
            server_name="0.0.0.0",
            server_port=7866,
            share=False
        )


if __name__ == "__main__":
    try:
        logger.info("🚀 Lancement de Seamless TTS...")
        app = SeamlessTTSApp()
        app.launch()
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        raise