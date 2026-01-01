#!/usr/bin/env python3
"""
SeamlessM4T S2ST - Speech-to-Speech Translation
Specialized speech translation with auto-GPU detection

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
import scipy.signal  # Pour le resampling audio

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeamlessM4T_S2ST:
    """Specialized speech translation (Speech-to-Speech)"""
    
    # Langues supportées par le modèle SeamlessM4T v2 pour S2ST
    S2ST_SUPPORTED_LANGUAGES = {
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
        logger.info("🔋 Initialisation de SeamlessM4T S2ST...")
        
        # Auto-détection GPU/CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🔥 Utilisation du device: {self.device}")
        
        # Chargement du modèle
        self._load_model()
    
    def _load_model(self):
        """Chargement optimisé du modèle"""
        try:
            logger.info("🔋 Chargement du modèle SeamlessM4T pour S2ST...")
            
            # Charger le processeur avec sampling_rate configuré dès le départ
            # pour éviter l'avertissement "sampling_rate not passed"
            try:
                # Créer le feature extractor avec le bon sampling_rate
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
                logger.info("✅ Processeur chargé avec sampling_rate=16000Hz (pas d'avertissement)")
            except Exception as e:
                # Fallback au cas où cette méthode échoue
                logger.warning(f"⚠️  Impossible de configurer le sampling_rate dès le départ: {e}")
                self.processor = AutoProcessor.from_pretrained(
                    "facebook/seamless-m4t-v2-large",
                    use_fast=False
                )
                if hasattr(self.processor, 'feature_extractor'):
                    self.processor.feature_extractor.sampling_rate = 16000
                    logger.info("✅ Feature extractor configuré pour 16000Hz (méthode alternative)")
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
    
    def translate_speech(self, audio_path: str, src_lang: str, tgt_lang: str) -> str:
        """Speech translation (S2ST) with long audio file handling"""
        try:
            # Validation
            if src_lang not in self.ALL_SUPPORTED_LANGUAGES:
                raise ValueError(f"Langue source non supportée: {src_lang}")
            if tgt_lang not in self.S2ST_SUPPORTED_LANGUAGES:
                supported_s2st_langs = ", ".join(self.S2ST_SUPPORTED_LANGUAGES.keys())
                raise ValueError(f"Langue cible non supportée pour S2ST: {tgt_lang}. "
                               f"Langues supportées: {supported_s2st_langs}")
            
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
                audio_paths = []
                
                for i, segment in enumerate(segments):
                    logger.info(f"Traitement segment {i+1}/{len(segments)}")
                    
                    # Sauvegarder le segment temporaire
                    segment_path = f"temp_audio/segment_{i}.wav"
                    os.makedirs("temp_audio", exist_ok=True)
                    sf.write(segment_path, segment, sample_rate)
                    
                    # Traduire le segment
                    try:
                        segment_output = self._translate_single_audio(segment_path, src_lang, tgt_lang)
                        audio_paths.append(segment_output)
                    except Exception as e:
                        logger.error(f"❌ Erreur de traduction du segment {i+1}: {e}")
                        # Nettoyer les segments déjà traités
                        for path in audio_paths:
                            if os.path.exists(path):
                                os.remove(path)
                        raise
                    finally:
                        # Nettoyer le segment temporaire
                        if os.path.exists(segment_path):
                            os.remove(segment_path)
                
                # Concatenation des segments audio traduits
                return self._concatenate_audios(audio_paths)
            else:
                # Traitement normal pour les audios courts
                return self._translate_single_audio(audio_path, src_lang, tgt_lang)
            
        except Exception as e:
            logger.error(f"❌ Erreur de traduction: {e}")
            raise
    
    def _translate_single_audio(self, audio_path: str, src_lang: str, tgt_lang: str) -> str:
        """Translation of a single audio file"""
        try:
            # Charger et traiter l'audio
            audio_input, sample_rate = sf.read(audio_path)
            
            # Détection du débit de parole (pour avertir l'utilisateur)
            duration = len(audio_input) / sample_rate
            
            # Calculer l'énergie moyenne pour détecter les pauses
            energy = np.sum(audio_input**2) / len(audio_input)
            
            # Si l'audio est court mais avec beaucoup d'énergie, avertir
            if duration < 5.0 and energy > 0.01:  # Audio court mais intense
                logger.warning(f"⚠️  L'audio semble avoir été dicté rapidement ({duration:.1f}s). La qualité de traduction pourrait être affectée.")
            
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
            
            # Générer la traduction
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
                            raise ValueError(f"Langue cible '{tgt_lang}' non supportée pour S2ST. "
                                           f"Langues supportées: {supported_langs}")
                    raise
            
            # Extraire l'audio de sortie - SeamlessM4T v2 retourne un tuple
            # output est un tuple où le premier élément contient les valeurs audio
            if isinstance(output, tuple):
                audio_values = output[0].cpu().numpy().squeeze()
            else:
                # Fallback pour d'autres versions
                audio_values = output["audio_values"][0].cpu().numpy().squeeze()
            
            # Conversion au format compatible
            if audio_values.dtype == np.float16:
                audio_values = audio_values.astype(np.float32)
            
            # Vérification de la sortie audio
            if len(audio_values) == 0:
                raise ValueError("Aucun audio généré par le modèle")
            
            if np.any(np.isnan(audio_values)):
                raise ValueError("L'audio généré contient des valeurs NaN")
            
            # Sauvegarde
            return self._save_audio(audio_values)
            
        except Exception as e:
            logger.error(f"❌ Erreur de traduction audio: {e}")
            raise
    
    def _concatenate_audios(self, audio_paths: list) -> str:
        """Concatenation of multiple audio files"""
        try:
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
            final_output_path = f"s2st_output/s2st_concatenated_{timestamp}.wav"
            os.makedirs("s2st_output", exist_ok=True)
            sf.write(final_output_path, concatenated_audio, first_sample_rate)
            
            # Nettoyage des fichiers temporaires
            for path in audio_paths:
                if os.path.exists(path):
                    os.remove(path)
            
            return final_output_path
            
        except Exception as e:
            logger.error(f"❌ Erreur de concatenation: {e}")
            raise
    
    def _save_audio(self, audio: np.ndarray) -> str:
        """Save the translated audio"""
        os.makedirs("s2st_output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"s2st_output/s2st_{timestamp}.wav"
        sf.write(output_path, audio, 16000)
        return output_path


class SeamlessS2STApp:
    """Gradio interface for SeamlessM4T S2ST"""
    
    def __init__(self):
        self.s2st = SeamlessM4T_S2ST()
        self.languages = self.s2st.S2ST_SUPPORTED_LANGUAGES
    
    def s2st_interface(self, audio_path: str, src_lang: str, tgt_lang: str) -> str:
        """Interface for speech translation"""
        try:
            output_path = self.s2st.translate_speech(audio_path, src_lang, tgt_lang)
            return output_path
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
        
        with gr.Blocks(title="SeamlessM4T S2ST") as app:
            gr.Markdown("""
            # 🎤 SeamlessM4T Speech-to-Speech Translation (S2ST)
            Specialized speech translation with auto-GPU detection
            
            **Fonctionnalités:**
            - 🎤 Audio vers Audio (S2ST)
            - 🔥 Auto-détection GPU/CPU
            - 🌍 Support multilingue (36 langues)
            - 💾 Sauvegarde automatique
            - ⏱️  Long audio handling (automatic segmentation)
            
            **Langues supportées pour S2ST:** Arabic, Bengali, Catalan, Czech, Mandarin, Welsh, Danish, German, English, Estonian, Finnish, French, Hindi, Indonesian, Italian, Japanese, Kannada, Korean, Maltese, Dutch, Persian, Polish, Portuguese, Romanian, Russian, Slovak, Spanish, Swedish, Swahili, Tamil, Telugu, Tagalog, Thai, Turkish, Ukrainian, Urdu, Uzbek, Vietnamese
            
            """)
            
            with gr.Row():
                s2st_audio = gr.Audio(
                    label="Audio to translate",
                    type="filepath",
                    sources=["microphone", "upload"]
                )
            
            with gr.Row():
                s2st_src_lang = gr.Dropdown(
                    choices=list(self.languages.keys()),
                    value="fra",
                    label="Source Language"
                )
                s2st_tgt_lang = gr.Dropdown(
                    choices=list(self.languages.keys()),
                    value="eng",
                    label="Target Language"
                )
            
            s2st_btn = gr.Button("Translate audio", variant="primary")
            s2st_output = gr.Audio(label="Translated audio", type="filepath")
            
            s2st_btn.click(
                fn=self.s2st_interface,
                inputs=[s2st_audio, s2st_src_lang, s2st_tgt_lang],
                outputs=s2st_output
            )
            
            gr.Markdown("""
            ---
            ### Informations
            - **Device:** " + ("🔥 GPU" if torch.cuda.is_available() else "❄️ CPU") + ""
            - **Modèle:** facebook/seamless-m4t-v2-large
            - **Échantillonnage:** 16kHz
            - **Format:** WAV
            - **Durée max par segment:** 30 secondes
            
            © 2024 SeamlessM4T S2ST
            """)
        
        return app
    
    def launch(self):
        """Lance l'application"""
        app = self.create_interface()
        app.launch(
            server_name="0.0.0.0",
            server_port=7867,  # Port différent de TTS
            share=False
        )


if __name__ == "__main__":
    try:
        logger.info("🚀 Lancement de SeamlessM4T S2ST...")
        app = SeamlessS2STApp()
        app.launch()
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        raise