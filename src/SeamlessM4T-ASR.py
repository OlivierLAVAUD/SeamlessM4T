#!/usr/bin/env python3
"""
SeamlessM4T ASR - Automatic Speech Recognition
Specialized speech recognition with auto-GPU detection

Based on: https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2
"""

import gradio as gr
import torch
import logging
from transformers import AutoProcessor, SeamlessM4Tv2Model
import numpy as np
import soundfile as sf
import os
from datetime import datetime
import scipy.signal  # Pour le resampling audio

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeamlessM4T_ASR:
    """Specialized speech recognition (Automatic Speech Recognition)"""
    
    # Langues supportées par le modèle SeamlessM4T v2 pour ASR
    ASR_SUPPORTED_LANGUAGES = {
        "afr": "Afrikaans", "amh": "Amharic", "ara": "Arabic", "arb": "Arabic (Modern Standard)",
        "asm": "Assamese", "aze": "Azerbaijani", "bel": "Belarusian", "ben": "Bengali",
        "bos": "Bosnian", "bul": "Bulgarian", "cat": "Catalan", "ceb": "Cebuano",
        "ces": "Czech", "ckb": "Central Kurdish", "cmn": "Mandarin Chinese", "cym": "Welsh",
        "dan": "Danish", "deu": "German", "ell": "Greek", "eng": "English",
        "est": "Estonian", "eus": "Basque", "fin": "Finnish", "fra": "French",
        "gaz": "West Central Oromo", "gle": "Irish", "glg": "Galician", "guj": "Gujarati",
        "heb": "Hebrew", "hin": "Hindi", "hrv": "Croatian", "hun": "Hungarian",
        "hye": "Armenian", "ibo": "Igbo", "ind": "Indonesian", "isl": "Icelandic",
        "ita": "Italian", "jav": "Javanese", "jpn": "Japanese", "kan": "Kannada",
        "kat": "Georgian", "kaz": "Kazakh", "kea": "Kabuverdianu", "khk": "Halh Mongolian",
        "khm": "Khmer", "kir": "Kyrgyz", "kor": "Korean", "lao": "Lao",
        "lit": "Lithuanian", "ltz": "Luxembourgish", "lug": "Ganda", "luo": "Luo",
        "lvs": "Standard Latvian", "mai": "Maithili", "mal": "Malayalam", "mar": "Marathi",
        "mkd": "Macedonian", "mlt": "Maltese", "mni": "Meitei", "mya": "Burmese",
        "nld": "Dutch", "nno": "Norwegian Nynorsk", "nob": "Norwegian Bokmål",
        "npi": "Nepali", "nya": "Nyanja", "oci": "Occitan", "ory": "Odia",
        "pan": "Punjabi", "pbt": "Southern Pashto", "pes": "Western Persian",
        "pol": "Polish", "por": "Portuguese", "ron": "Romanian", "rus": "Russian",
        "slk": "Slovak", "slv": "Slovenian", "sna": "Shona", "snd": "Sindhi",
        "som": "Somali", "spa": "Spanish", "srp": "Serbian", "swe": "Swedish",
        "swh": "Swahili", "tam": "Tamil", "tel": "Telugu", "tgk": "Tajik",
        "tgl": "Tagalog", "tha": "Thai", "tur": "Turkish", "ukr": "Ukrainian",
        "urd": "Urdu", "uzb": "Uzbek", "uzn": "Northern Uzbek", "vie": "Vietnamese",
        "xho": "Xhosa", "yid": "Yiddish", "yor": "Yoruba", "yue": "Cantonese",
        "zho": "Chinese", "zsm": "Standard Malay", "zul": "Zulu"
    }
    
    def __init__(self):
        """Initialisation avec auto-détection GPU"""
        logger.info("🔋 Initialisation de SeamlessM4T ASR...")
        
        # Auto-détection GPU/CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🔥 Utilisation du device: {self.device}")
        
        # Chargement du modèle
        self._load_model()
    
    def _load_model(self):
        """Chargement optimisé du modèle"""
        try:
            logger.info("🔋 Chargement du modèle SeamlessM4T pour ASR...")
            
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
    
    def recognize_speech(self, audio_path: str, src_lang: str) -> str:
        """Speech recognition (ASR) with long audio file handling"""
        try:
            # Validation
            if src_lang not in self.ASR_SUPPORTED_LANGUAGES:
                raise ValueError(f"Langue non supportée pour ASR: {src_lang}")
            
            # Charger l'audio
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Fichier audio non trouvé: {audio_path}")
            
            audio_input, sample_rate = sf.read(audio_path)
            
            # Vérification de la durée (limite pour éviter les problèmes mémoire)
            max_duration = 30  # secondes
            if len(audio_input) / sample_rate > max_duration:
                logger.warning(f"Audio trop long ({len(audio_input)/sample_rate:.1f}s), découpage en segments...")
                # Découper l'audio en segments
                segment_length = sample_rate * max_duration  # 30 secondes
                segments = [audio_input[i:i+segment_length] for i in range(0, len(audio_input), segment_length)]
                
                recognized_texts = []
                
                for i, segment in enumerate(segments):
                    logger.info(f"Traitement segment {i+1}/{len(segments)}")
                    
                    # Sauvegarder le segment temporaire
                    segment_path = f"temp_audio/segment_{i}.wav"
                    os.makedirs("temp_audio", exist_ok=True)
                    sf.write(segment_path, segment, sample_rate)
                    
                    # Reconnaître le segment
                    try:
                        segment_text = self._recognize_single_audio(segment_path, src_lang)
                        recognized_texts.append(segment_text)
                    except Exception as e:
                        logger.error(f"❌ Erreur de reconnaissance du segment {i+1}: {e}")
                        raise
                    finally:
                        # Nettoyer le segment temporaire
                        if os.path.exists(segment_path):
                            os.remove(segment_path)
                
                # Combiner les textes reconnus
                return " ".join(recognized_texts)
            else:
                # Traitement normal pour les audios courts
                return self._recognize_single_audio(audio_path, src_lang)
            
        except Exception as e:
            logger.error(f"❌ Erreur de reconnaissance: {e}")
            raise
    
    def _recognize_single_audio(self, audio_path: str, src_lang: str) -> str:
        """Recognition of a single audio file"""
        try:
            # Charger et traiter l'audio
            audio_input, sample_rate = sf.read(audio_path)
            
            # Détection du débit de parole (pour avertir l'utilisateur)
            duration = len(audio_input) / sample_rate
            
            # Calculer l'énergie moyenne pour détecter les pauses
            energy = np.sum(audio_input**2) / len(audio_input)
            
            # Si l'audio est court mais avec beaucoup d'énergie, avertir
            if duration < 5.0 and energy > 0.01:  # Audio court mais intense
                logger.warning(f"⚠️  L'audio semble avoir été dicté rapidement ({duration:.1f}s). La qualité de reconnaissance pourrait être affectée.")
            
            # Conversion et validation de l'audio
            if len(audio_input.shape) > 1:
                # Convertir stéréo en mono si nécessaire
                audio_input = np.mean(audio_input, axis=1)
            
            # Assurer que l'audio est au bon format
            if audio_input.dtype != np.float32:
                audio_input = audio_input.astype(np.float32)
            
            # Normalisation (optionnelle mais recommandée)
            audio_input = audio_input / np.max(np.abs(audio_input))
            
            # Resampling si nécessaire (SeamlessM4T attend 16kHz)
            if sample_rate != 16000:
                try:
                    # Utiliser une méthode simple de resampling
                    original_length = len(audio_input)
                    target_length = int(original_length * 16000 / sample_rate)
                    audio_input = scipy.signal.resample(audio_input, target_length)
                    logger.info(f"Resampling de {sample_rate}Hz à 16000Hz")
                except Exception as e:
                    logger.error(f"❌ Erreur de resampling: {e}")
                    raise ValueError(f"Format audio non supporté: {sample_rate}Hz. SeamlessM4T nécessite 16kHz.")
            
            # Vérification finale avant traitement
            if len(audio_input) == 0:
                raise ValueError("Fichier audio vide ou non valide")
            
            if np.any(np.isnan(audio_input)):
                raise ValueError("Fichier audio contient des valeurs NaN")
            
            # Préparer les entrées pour le modèle
            try:
                inputs = self.processor(
                    audios=[audio_input],
                    src_lang=src_lang,
                    return_tensors="pt"
                ).to(self.device)
            except Exception as e:
                logger.error(f"❌ Erreur de traitement audio: {e}")
                raise ValueError(f"Format audio non compatible: {str(e)}")
            
            # Générer la reconnaissance
            with torch.no_grad():
                try:
                    output = self.model.generate(**inputs, tgt_lang=src_lang, task="asr")
                except Exception as model_error:
                    # Gestion des erreurs spécifiques du modèle
                    error_msg = str(model_error)
                    if "not supported by this model" in error_msg:
                        # Extraire la liste des langues supportées de l'erreur
                        start_idx = error_msg.find("in ") + 3
                        end_idx = error_msg.find(". Note that")
                        if start_idx > 0 and end_idx > 0:
                            supported_langs = error_msg[start_idx:end_idx].strip()
                            raise ValueError(f"Langue '{src_lang}' non supportée pour ASR. "
                                           f"Langues supportées: {supported_langs}")
                    raise
            
            # Extraire le texte reconnu
            # Pour ASR, le modèle retourne directement le texte
            if isinstance(output, dict) and "text" in output:
                recognized_text = output["text"][0]
            else:
                # Méthode alternative pour extraire le texte
                recognized_text = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Vérification du texte reconnu
            if not recognized_text or len(recognized_text.strip()) == 0:
                raise ValueError("Aucun texte reconnu par le modèle")
            
            return recognized_text
            
        except Exception as e:
            logger.error(f"❌ Erreur de reconnaissance audio: {e}")
            raise


class SeamlessASRApp:
    """Gradio interface for SeamlessM4T ASR"""
    
    def __init__(self):
        self.asr = SeamlessM4T_ASR()
        self.languages = self.asr.ASR_SUPPORTED_LANGUAGES
    
    def asr_interface(self, audio_path: str, src_lang: str) -> str:
        """Interface for speech recognition"""
        try:
            recognized_text = self.asr.recognize_speech(audio_path, src_lang)
            return recognized_text
        except ValueError as e:
            # Erreur de validation des langues - limiter la longueur du message
            error_msg = str(e)
            if len(error_msg) > 200:
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
        """Create the Gradio interface"""
        
        with gr.Blocks(title="SeamlessM4T ASR") as app:
            gr.Markdown("""
            # 🎤 SeamlessM4T Automatic Speech Recognition (ASR)
            Specialized speech recognition with auto-GPU detection
            
            **Fonctionnalités:**
            - 🎤 Audio vers Texte (ASR)
            - 🔥 Auto-détection GPU/CPU
            - 🌍 Support multilingue (100+ langues)
            - 💾 Sauvegarde automatique
            - ⏱️  Long audio handling (automatic segmentation)
            
            **Langues supportées pour ASR:** 100+ langues incluant Français, Anglais, Espagnol, Allemand, Chinois, Arabe, etc.
            
            """)
            
            with gr.Row():
                asr_audio = gr.Audio(
                    label="Audio à reconnaître",
                    type="filepath",
                    sources=["microphone", "upload"]
                )
            
            with gr.Row():
                asr_src_lang = gr.Dropdown(
                    choices=list(self.languages.keys()),
                    value="fra",
                    label="Source Language"
                )
            
            asr_btn = gr.Button("Reconnaître la parole", variant="primary")
            asr_output = gr.Textbox(label="Texte reconnu", lines=5)
            
            asr_btn.click(
                fn=self.asr_interface,
                inputs=[asr_audio, asr_src_lang],
                outputs=asr_output
            )
            
            gr.Markdown("""
            ---
            ### Informations
            - **Device:** " + ("🔥 GPU" if torch.cuda.is_available() else "❄️ CPU") + ""
            - **Modèle:** facebook/seamless-m4t-v2-large
            - **Échantillonnage:** 16kHz
            - **Format:** WAV
            - **Durée max par segment:** 30 secondes
            
            © 2024 SeamlessM4T ASR
            """)
        
        return app
    
    def launch(self):
        """Lance l'application"""
        app = self.create_interface()
        app.launch(
            server_name="0.0.0.0",
            server_port=7868,  # Port différent pour ASR
            share=False
        )


if __name__ == "__main__":
    try:
        logger.info("🚀 Lancement de SeamlessM4T ASR...")
        app = SeamlessASRApp()
        app.launch()
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        raise