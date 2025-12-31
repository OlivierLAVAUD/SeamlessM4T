#!/usr/bin/env python3
"""
SeamlessM4T T2TT - Text-to-Text Translation
Traduction textuelle spécialisée avec auto-détection GPU

Basé sur: https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2
"""

import gradio as gr
import torch
import logging
from transformers import AutoProcessor, SeamlessM4Tv2Model
import numpy as np
import os
from datetime import datetime

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeamlessM4T_T2TT:
    """Traduction textuelle (Text-to-Text)"""
    
    # Langues supportées par le modèle SeamlessM4T v2 pour T2TT
    # Selon la documentation, T2TT supporte les mêmes langues que S2TT
    T2TT_SUPPORTED_LANGUAGES = {
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
        logger.info("🔋 Initialisation de SeamlessM4T T2TT...")
        
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
            logger.info("🔋 Chargement du modèle SeamlessM4T pour T2TT...")
            
            # Créer le processeur pour T2TT
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
    
    def translate_text(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Traduction textuelle (T2TT) avec gestion des erreurs"""
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
            if tgt_lang not in self.T2TT_SUPPORTED_LANGUAGES:
                supported_t2tt_langs = ", ".join(self.T2TT_SUPPORTED_LANGUAGES.keys())
                raise ValueError(f"Langue cible non supportée pour T2TT: {tgt_lang}. "
                               f"Langues supportées: {supported_t2tt_langs}")
            
            # Validation du texte
            if not text or not text.strip():
                raise ValueError("Le texte à traduire ne peut pas être vide")
            
            if len(text) > 10000:  # Limite pour éviter les problèmes mémoire
                logger.warning(f"⚠️  Texte très long ({len(text)} caractères). Découpage en segments...")
                # Découper le texte en segments plus petits
                segment_length = 5000  # Environ 5000 caractères par segment
                segments = [text[i:i+segment_length] for i in range(0, len(text), segment_length)]
                translated_segments = []
                
                for i, segment in enumerate(segments):
                    logger.info(f"Traitement segment {i+1}/{len(segments)}")
                    try:
                        segment_translation = self._translate_single_text(segment, src_lang, tgt_lang)
                        translated_segments.append(segment_translation)
                    except Exception as segment_error:
                        logger.error(f"❌ Erreur de traduction du segment {i+1}: {segment_error}")
                        translated_segments.append(f"[ERREUR: {str(segment_error)}]")
                
                # Concatenation des traductions
                return " ".join(translated_segments)
            else:
                # Traitement normal pour les textes courts
                return self._translate_single_text(text, src_lang, tgt_lang)
            
        except Exception as e:
            logger.error(f"❌ Erreur de traduction: {e}")
            # Toujours vider la mémoire GPU en cas d'erreur
            self._cleanup_gpu_memory()
            raise
    
    def _translate_single_text(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Traduction d'un seul texte"""
        try:
            # Préparer les entrées pour le modèle (T2TT utilise text_inputs)
            inputs = self.processor(
                text=[text],
                src_lang=src_lang,
                return_tensors="pt"
            ).to(self.device)
            
            # Générer la traduction
            with torch.no_grad():
                try:
                    # Pour T2TT avec SeamlessM4T v2, nous devons utiliser la bonne approche
                    # Selon la documentation, pour T2TT nous devons:
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
                            raise ValueError(f"Langue cible '{tgt_lang}' non supportée pour T2TT. "
                                           f"Langues supportées: {supported_langs}")
                    raise
            
            # Extraire le texte généré pour T2TT
            # Selon la documentation SeamlessM4T v2, la sortie est un objet spécifique
            
            # Pour T2TT, la sortie devrait être un objet avec les tokens de texte
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
                
                logger.info(f"Texte traduit: {text}")
                return text
            except Exception as e:
                logger.error(f"Erreur de décodage: {e}")
                raise ValueError(f"Impossible de décoder la sortie: {e}")
            finally:
                # Toujours vider la mémoire GPU après le traitement
                if self.device == "cuda":
                    self._cleanup_gpu_memory()
                    
        except Exception as e:
            logger.error(f"❌ Erreur de traduction textuelle: {e}")
            raise


class SeamlessT2TTApp:
    """Interface Gradio pour SeamlessM4T T2TT"""
    
    def __init__(self):
        self.t2tt = SeamlessM4T_T2TT()
        self.languages = self.t2tt.T2TT_SUPPORTED_LANGUAGES
    
    def t2tt_interface(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Interface pour la traduction textuelle"""
        try:
            translation_output = self.t2tt.translate_text(text, src_lang, tgt_lang)
            return translation_output
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
            # Toujours vider la mémoire GPU après chaque requête
            if hasattr(self.t2tt, '_cleanup_gpu_memory'):
                self.t2tt._cleanup_gpu_memory()
    
    def create_interface(self):
        """Crée l'interface Gradio"""
        
        with gr.Blocks(title="SeamlessM4T T2TT") as app:
            gr.Markdown("""
            # 📝 SeamlessM4T Text-to-Text Translation (T2TT)
            Traduction textuelle spécialisée avec auto-détection GPU
            
            **Fonctionnalités:**
            - 📝 Texte vers Texte (T2TT)
            - 🔥 Auto-détection GPU/CPU
            - 🌍 Support multilingue (36 langues)
            - 📄 Gestion des textes longs (découpage automatique)
            - 🧹 Nettoyage automatique de la mémoire GPU
            
            **Langues supportées pour T2TT:** Arabic, Bengali, Catalan, Czech, Mandarin, Welsh, Danish, German, English, Estonian, Finnish, French, Hindi, Indonesian, Italian, Japanese, Kannada, Korean, Maltese, Dutch, Persian, Polish, Portuguese, Romanian, Russian, Slovak, Spanish, Swedish, Swahili, Tamil, Telugu, Tagalog, Thai, Turkish, Ukrainian, Urdu, Uzbek, Vietnamese
            
            """)
            
            with gr.Row():
                t2tt_text = gr.Textbox(
                    label="Texte à traduire",
                    lines=5,
                    placeholder="Entrez le texte à traduire ici..."
                )
            
            with gr.Row():
                t2tt_src_lang = gr.Dropdown(
                    choices=list(self.languages.keys()),
                    value="fra",
                    label="Langue source"
                )
                t2tt_tgt_lang = gr.Dropdown(
                    choices=list(self.languages.keys()),
                    value="eng",
                    label="Langue cible"
                )
            
            t2tt_btn = gr.Button("Traduire texte", variant="primary")
            t2tt_output = gr.Textbox(label="Texte traduit", lines=5)
            
            t2tt_btn.click(
                fn=self.t2tt_interface,
                inputs=[t2tt_text, t2tt_src_lang, t2tt_tgt_lang],
                outputs=t2tt_output
            )
            
            gr.Markdown("""
            ---
            ### Informations
            - **Device:** " + ("🔥 GPU" if torch.cuda.is_available() else "❄️ CPU") + ""
            - **Modèle:** facebook/seamless-m4t-v2-large
            - **Durée max par segment:** 5000 caractères
            - **Nettoyage GPU:** Après 5 requêtes
            
            © 2024 SeamlessM4T T2TT
            """)
        
        return app
    
    def launch(self):
        """Lance l'application"""
        try:
            app = self.create_interface()
            app.launch(
                server_name="0.0.0.0",
                server_port=7869,  # Port différent des autres applications
                share=False
            )
        finally:
            # Nettoyer les ressources GPU à la fin
            if hasattr(self.t2tt, '_cleanup_gpu_memory'):
                self.t2tt._cleanup_gpu_memory()
            logger.info("🧹 Ressources GPU nettoyées à la fin de l'application")


if __name__ == "__main__":
    try:
        logger.info("🚀 Lancement de SeamlessM4T T2TT...")
        app = SeamlessT2TTApp()
        app.launch()
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        # Nettoyer les ressources GPU en cas d'erreur fatale
        if 'app' in locals() and hasattr(app.t2tt, '_cleanup_gpu_memory'):
            app.t2tt._cleanup_gpu_memory()
        raise