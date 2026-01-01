#!/usr/bin/env python3
"""
SeamlessM4T S2TT - Speech-to-Text Translation
Speech transcription and translation with auto-GPU detection

Based on: https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2
"""

import gradio as gr
import torch
import logging
from transformers import AutoProcessor, SeamlessM4Tv2Model, SeamlessM4TFeatureExtractor
import numpy as np
import soundfile as sf
import os
from datetime import datetime
import scipy.signal

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeamlessM4T_S2TT:
    """Speech transcription and translation (Speech-to-Text)"""
    
    # Langues supportées par le modèle SeamlessM4T v2 pour S2TT
    S2TT_SUPPORTED_LANGUAGES = {
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
        logger.info("🔋 Initialisation de SeamlessM4T S2TT...")
        
        # Auto-détection GPU/CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🔥 Utilisation du device: {self.device}")
        
        # Chargement du modèle
        self._load_model()
        
        # Initialiser le compteur de requêtes pour la gestion mémoire
        self.request_count = 0
        self.MAX_REQUESTS_BEFORE_CLEANUP = 5  # Nettoyer après 5 requêtes
    
    def _cleanup_gpu_memory(self):
        """Nettoyer la mémoire GPU de manière sûre"""
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                # Vider le cache CUDA
                torch.cuda.empty_cache()
                
                # Synchroniser pour s'assurer que le nettoyage est terminé
                torch.cuda.synchronize()
                
                # Vérifier la mémoire disponible
                free_memory = torch.cuda.mem_get_info()[0] / 1024**3  # en Go
                logger.info(f"🧹 Mémoire GPU nettoyée. Disponible: {free_memory:.1f} Go")
                
                return True
        except Exception as e:
            logger.warning(f"⚠️  Impossible de nettoyer la mémoire GPU: {e}")
        return False
    
    def _load_model(self):
        """Chargement optimisé du modèle sans avertissement sampling_rate"""
        try:
            logger.info("🔋 Chargement du modèle SeamlessM4T pour S2TT...")
            
            # Créer le feature extractor avec sampling_rate explicite
            feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
                "facebook/seamless-m4t-v2-large",
                sampling_rate=16000
            )
            
            # Créer le processeur avec ce feature extractor configuré
            self.processor = AutoProcessor.from_pretrained(
                "facebook/seamless-m4t-v2-large",
                feature_extractor=feature_extractor,
                use_fast=False
            )
            logger.info("✅ Processeur chargé sans avertissement sampling_rate")
            
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
    
    def transcribe_speech(self, audio_path: str, src_lang: str, tgt_lang: str) -> str:
        """Speech transcription and translation (S2TT) with long audio handling"""
        try:
            # Incrémenter le compteur de requêtes
            self.request_count += 1
            
            # Nettoyer la mémoire GPU périodiquement
            if self.device == "cuda" and self.request_count >= self.MAX_REQUESTS_BEFORE_CLEANUP:
                self._cleanup_gpu_memory()
                self.request_count = 0  # Réinitialiser le compteur
            
            # Validation
            if src_lang not in self.ALL_SUPPORTED_LANGUAGES:
                raise ValueError(f"Langue source non supportée: {src_lang}")
            if tgt_lang not in self.S2TT_SUPPORTED_LANGUAGES:
                supported_s2tt_langs = ", ".join(self.S2TT_SUPPORTED_LANGUAGES.keys())
                raise ValueError(f"Langue cible non supportée pour S2TT: {tgt_lang}. "
                               f"Langues supportées: {supported_s2tt_langs}")
            
            # Charger l'audio
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Fichier audio non trouvé: {audio_path}")
            
            audio_input, sample_rate = sf.read(audio_path)
            
            # Détection du débit de parole
            duration = len(audio_input) / sample_rate
            energy = np.sum(audio_input**2) / len(audio_input)
            
            if duration < 5.0 and energy > 0.01:
                logger.warning(f"⚠️  L'audio semble avoir été dicté rapidement ({duration:.1f}s)")
            
            # Vérification de la durée (limite pour éviter les problèmes mémoire)
            max_duration = 60  # secondes pour S2TT (plus long que S2ST)
            if len(audio_input) / sample_rate > max_duration:
                logger.warning(f"Audio trop long ({duration:.1f}s), découpage en segments...")
                # Découper l'audio en segments
                segment_length = sample_rate * max_duration
                segments = [audio_input[i:i+segment_length] for i in range(0, len(audio_input), segment_length)]
                text_segments = []
                
                for i, segment in enumerate(segments):
                    logger.info(f"Traitement segment {i+1}/{len(segments)}")
                    
                    # Sauvegarder le segment temporaire
                    segment_path = f"temp_audio/segment_{i}.wav"
                    os.makedirs("temp_audio", exist_ok=True)
                    sf.write(segment_path, segment, sample_rate)
                    
                    # Transcrire le segment
                    try:
                        segment_text = self._transcribe_single_audio(segment_path, src_lang, tgt_lang)
                        text_segments.append(segment_text)
                    finally:
                        # Nettoyer le segment temporaire
                        if os.path.exists(segment_path):
                            os.remove(segment_path)
                
                # Concatenation des textes
                return " ".join(text_segments)
            else:
                # Traitement normal pour les audios courts
                return self._transcribe_single_audio(audio_path, src_lang, tgt_lang)
            
        except Exception as e:
            logger.error(f"❌ Erreur de transcription: {e}")
            raise
    
    def _transcribe_single_audio(self, audio_path: str, src_lang: str, tgt_lang: str) -> str:
        """Transcription d'un seul fichier audio"""
        try:
            # Charger et traiter l'audio
            audio_input, sample_rate = sf.read(audio_path)
            
            # Conversion et validation de l'audio
            if len(audio_input.shape) > 1:
                audio_input = np.mean(audio_input, axis=1)
            
            if audio_input.dtype != np.float32:
                audio_input = audio_input.astype(np.float32)
            
            # Normalisation
            audio_input = audio_input / np.max(np.abs(audio_input))
            
            # Resampling si nécessaire
            if sample_rate != 16000:
                try:
                    original_length = len(audio_input)
                    target_length = int(original_length * 16000 / sample_rate)
                    audio_input = scipy.signal.resample(audio_input, target_length)
                    logger.info(f"Resampling de {sample_rate}Hz à 16000Hz")
                except Exception as e:
                    logger.error(f"❌ Erreur de resampling: {e}")
                    raise ValueError(f"Format audio non supporté: {sample_rate}Hz")
            
            # Vérification finale
            if len(audio_input) == 0:
                raise ValueError("Fichier audio vide ou non valide")
            
            if np.any(np.isnan(audio_input)):
                raise ValueError("Fichier audio contient des valeurs NaN")
            
            # Préparer les entrées pour le modèle (S2TT utilise text_targets)
            inputs = self.processor(
                audios=[audio_input],
                src_lang=src_lang,
                return_tensors="pt"
            ).to(self.device)
            
            # Générer la transcription
            with torch.no_grad():
                try:
                    # Pour S2TT avec SeamlessM4T v2, nous devons utiliser la bonne approche
                    # Selon la documentation, pour S2TT nous devons:
                    # 1. Désactiver la génération audio
                    # 2. Utiliser la tâche appropriée
                    output = self.model.generate(
                        **inputs,
                        tgt_lang=tgt_lang,
                        generate_speech=False  # Désactiver la génération audio
                    )
                except Exception as model_error:
                    error_msg = str(model_error)
                    if "not supported by this model" in error_msg:
                        start_idx = error_msg.find("in ") + 3
                        end_idx = error_msg.find(". Note that")
                        if start_idx > 0 and end_idx > 0:
                            supported_langs = error_msg[start_idx:end_idx].strip()
                            raise ValueError(f"Langue cible '{tgt_lang}' non supportée pour S2TT. "
                                           f"Langues supportées: {supported_langs}")
                    raise
            
            # Extraire le texte généré pour S2TT
            # Selon la documentation SeamlessM4T v2, la sortie est un objet spécifique
            
            # Vérifier le type de sortie
            logger.info(f"Type de sortie du modèle: {type(output)}")
            
            # Pour S2TT, la sortie devrait être un objet avec les tokens de texte
            if hasattr(output, 'sequences'):
                # Cas 1: La sortie a un attribut 'sequences'
                generated_tokens = output.sequences
            elif isinstance(output, tuple):
                # Cas 2: La sortie est un tuple
                generated_tokens = output[0]
            elif hasattr(output, 'cpu'):
                # Cas 3: La sortie est un tensor
                generated_tokens = output.cpu()
            else:
                # Cas 4: Autre format
                generated_tokens = output
            
            # Convertir en texte
            try:
                # Utiliser batch_decode qui gère les différents formats
                text = self.processor.tokenizer.batch_decode(
                    generated_tokens,
                    skip_special_tokens=True
                )
                
                # Prendre le premier élément si c'est une liste
                if isinstance(text, list) and len(text) > 0:
                    text = text[0]
                elif isinstance(text, list):
                    text = ""
                
                logger.info(f"Texte généré: {text}")
                return text
            except Exception as e:
                logger.error(f"Erreur de décodage: {e}")
                logger.error(f"Type de generated_tokens: {type(generated_tokens)}")
                if hasattr(generated_tokens, 'shape'):
                    logger.error(f"Forme: {generated_tokens.shape}")
                raise ValueError(f"Impossible de décoder la sortie: {e}")
            finally:
                # Nettoyer la mémoire GPU après chaque transcription
                if self.device == "cuda":
                    self._cleanup_gpu_memory()
            
        except Exception as e:
            logger.error(f"❌ Erreur de transcription audio: {e}")
            raise


class SeamlessS2TTApp:
    """Gradio interface for SeamlessM4T S2TT"""
    
    def __init__(self):
        self.s2tt = SeamlessM4T_S2TT()
        self.languages = self.s2tt.S2TT_SUPPORTED_LANGUAGES
    
    def s2tt_interface(self, audio_path: str, src_lang: str, tgt_lang: str) -> str:
        """Interface for speech transcription"""
        try:
            text_output = self.s2tt.transcribe_speech(audio_path, src_lang, tgt_lang)
            return text_output
        except ValueError as e:
            error_msg = str(e)
            if len(error_msg) > 200:
                if "Langues supportées:" in error_msg:
                    start_idx = error_msg.find("Langues supportées:")
                    return error_msg[:start_idx + 20] + "... (voir logs)"
                else:
                    return error_msg[:200] + "..."
            return f"❌ {error_msg}"
        except Exception as e:
            error_msg = str(e)
            if len(error_msg) > 200:
                return f"❌ {error_msg[:200]}..."
            return f"❌ Erreur inattendue: {error_msg}"
        finally:
            # Nettoyer la mémoire GPU après chaque requête
            if hasattr(self.s2tt, '_cleanup_gpu_memory'):
                self.s2tt._cleanup_gpu_memory()
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        with gr.Blocks(title="SeamlessM4T S2TT") as app:
            gr.Markdown("""
            # 🎤 SeamlessM4T Speech-to-Text Translation (S2TT)
            Speech transcription and translation with auto-GPU detection
            
            **Fonctionnalités:**
            - 🎤 Audio vers Texte (S2TT)
            - 🔥 Auto-détection GPU/CPU
            - 🌍 Support multilingue (36 langues)
            - 📝 Transcription + traduction
            - ⏱️  Long audio handling (automatic segmentation)
            
            **Langues supportées pour S2TT:** Arabic, Bengali, Catalan, Czech, Mandarin, Welsh, Danish, German, English, Estonian, Finnish, French, Hindi, Indonesian, Italian, Japanese, Kannada, Korean, Maltese, Dutch, Persian, Polish, Portuguese, Romanian, Russian, Slovak, Spanish, Swedish, Swahili, Tamil, Telugu, Tagalog, Thai, Turkish, Ukrainian, Urdu, Uzbek, Vietnamese

            """)
            
            with gr.Row():
                s2tt_audio = gr.Audio(
                    label="Audio to transcribe",
                    type="filepath",
                    sources=["microphone", "upload"]
                )
            
            with gr.Row():
                s2tt_src_lang = gr.Dropdown(
                    choices=list(self.languages.keys()),
                    value="fra",
                    label="Source Language"
                )
                s2tt_tgt_lang = gr.Dropdown(
                    choices=list(self.languages.keys()),
                    value="eng",
                    label="Target Language"
                )
            
            s2tt_btn = gr.Button("Transcribe audio", variant="primary")
            s2tt_output = gr.Textbox(label="Transcribed text", lines=5)
            
            s2tt_btn.click(
                fn=self.s2tt_interface,
                inputs=[s2tt_audio, s2tt_src_lang, s2tt_tgt_lang],
                outputs=s2tt_output
            )
            
            gr.Markdown("""
            ---
            ### Informations
            - **Device:** " + ("🔥 GPU" if torch.cuda.is_available() else "❄️ CPU") + ""
            - **Modèle:** facebook/seamless-m4t-v2-large
            - **Échantillonnage:** 16kHz
            - **Durée max par segment:** 60 secondes
            
            © 2024 SeamlessM4T S2TT
            """)
        
        return app
    
    def launch(self):
        """Lance l'application"""
        try:
            app = self.create_interface()
            app.launch(
                server_name="0.0.0.0",
                server_port=7868,  # Port différent de TTS et S2ST
                share=False
            )
        finally:
            # Nettoyer les ressources GPU à la fin
            if hasattr(self.s2tt, '_cleanup_gpu_memory'):
                self.s2tt._cleanup_gpu_memory()
            logger.info("🧹 Ressources GPU nettoyées à la fin de l'application")


if __name__ == "__main__":
    try:
        logger.info("🚀 Lancement de SeamlessM4T S2TT...")
        app = SeamlessS2TTApp()
        app.launch()
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        # Nettoyer les ressources GPU en cas d'erreur fatale
        if 'app' in locals() and hasattr(app.s2tt, '_cleanup_gpu_memory'):
            app.s2tt._cleanup_gpu_memory()
        raise