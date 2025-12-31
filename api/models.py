#!/usr/bin/env python3
"""
Modèles Pydantic pour l'API SeamlessM4T
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class LanguageCode(str, Enum):
    """Codes des langues supportées"""
    arb = "arb"  # Arabic
    ben = "ben"  # Bengali
    cat = "cat"  # Catalan
    ces = "ces"  # Czech
    cmn = "cmn"  # Mandarin Chinese
    cym = "cym"  # Welsh
    dan = "dan"  # Danish
    deu = "deu"  # German
    eng = "eng"  # English
    est = "est"  # Estonian
    fin = "fin"  # Finnish
    fra = "fra"  # French
    hin = "hin"  # Hindi
    ind = "ind"  # Indonesian
    ita = "ita"  # Italian
    jpn = "jpn"  # Japanese
 #   kan = "kan"  # Kannada
    kor = "kor"  # Korean
    mlt = "mlt"  # Maltese
    nld = "nld"  # Dutch
    pes = "pes"  # Western Persian
    pol = "pol"  # Polish
    por = "por"  # Portuguese
    ron = "ron"  # Romanian
    rus = "rus"  # Russian
    slk = "slk"  # Slovak
    spa = "spa"  # Spanish
    swe = "swe"  # Swedish
    swh = "swh"  # Swahili
    tam = "tam"  # Tamil
    tel = "tel"  # Telugu
    tgl = "tgl"  # Tagalog
    tha = "tha"  # Thai
    tur = "tur"  # Turkish
    ukr = "ukr"  # Ukrainian
    urd = "urd"  # Urdu
    uzn = "uzn"  # Northern Uzbek
    vie = "vie"  # Vietnamese


class S2STRequest(BaseModel):
    """Requête pour Speech-to-Speech Translation"""
    src_lang: LanguageCode = Field(..., description="Langue source (code ISO 639-3)")
    tgt_lang: LanguageCode = Field(..., description="Langue cible (code ISO 639-3)")
    audio_format: Optional[str] = Field("wav", description="Format audio de sortie")


class S2TTRequest(BaseModel):
    """Requête pour Speech-to-Text Translation"""
    src_lang: LanguageCode = Field(..., description="Langue source (code ISO 639-3)")
    tgt_lang: LanguageCode = Field(..., description="Langue cible (code ISO 639-3)")


class T2STRequest(BaseModel):
    """Requête pour Text-to-Speech Translation"""
    text: str = Field(..., description="Texte à synthétiser", max_length=5000)
    src_lang: LanguageCode = Field(..., description="Langue source (code ISO 639-3)")
    tgt_lang: LanguageCode = Field(..., description="Langue cible (code ISO 639-3)")
    audio_format: Optional[str] = Field("wav", description="Format audio de sortie")


class T2TTRequest(BaseModel):
    """Requête pour Text-to-Text Translation"""
    text: str = Field(..., description="Texte à traduire", max_length=5000)
    src_lang: LanguageCode = Field(..., description="Langue source (code ISO 639-3)")
    tgt_lang: LanguageCode = Field(..., description="Langue cible (code ISO 639-3)")


class TranslationResponse(BaseModel):
    """Réponse générique pour les traductions"""
    success: bool = Field(..., description="Statut de la requête")
    message: Optional[str] = Field(None, description="Message d'état")
    result: Optional[str] = Field(None, description="Résultat de la traduction")
    error: Optional[str] = Field(None, description="Message d'erreur")


class AudioTranslationResponse(TranslationResponse):
    """Réponse pour les traductions audio"""
    audio_url: Optional[str] = Field(None, description="URL du fichier audio généré")
    duration: Optional[float] = Field(None, description="Durée de l'audio en secondes")


class TextTranslationResponse(TranslationResponse):
    """Réponse pour les traductions textuelles"""
    text: Optional[str] = Field(None, description="Texte traduit")
    character_count: Optional[int] = Field(None, description="Nombre de caractères")


class HealthCheckResponse(BaseModel):
    """Réponse pour le health check"""
    status: str = Field(..., description="Statut du service")
    model_loaded: bool = Field(..., description="Modèle chargé")
    device: str = Field(..., description="Device utilisé (CPU/GPU)")
    gpu_available: bool = Field(..., description="GPU disponible")
    gpu_memory: Optional[dict] = Field(None, description="Mémoire GPU")


class ErrorResponse(BaseModel):
    """Réponse d'erreur standard"""
    error: str = Field(..., description="Message d'erreur")
    detail: Optional[str] = Field(None, description="Détails de l'erreur")
    status_code: Optional[int] = Field(None, description="Code d'état HTTP")


class LanguageInfo(BaseModel):
    """Informations sur une langue"""
    code: str = Field(..., description="Code de la langue")
    name: str = Field(..., description="Nom complet de la langue")


class SupportedLanguagesResponse(BaseModel):
    """Réponse pour la liste des langues supportées"""
    languages: List[LanguageInfo] = Field(..., description="Liste des langues supportées")
    count: int = Field(..., description="Nombre de langues supportées")
