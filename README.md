# SeamlessM4T API v2



**API unifiée pour les services de traduction SeamlessM4T v2**

![SeamlessM4T Logo](https://raw.githubusercontent.com/facebookresearch/seamless_communication/main/docs/images/seamless_logo.png)




## 📋 Description

Cette API fournit une interface unifiée pour utiliser les capacités du modèle **SeamlessM4T v2** de Meta. Elle offre quatre services principaux de traduction et transcription multilingue :

- **Speech-to-Speech Translation (S2ST)** : Traduction vocale directe
- **Speech-to-Text Translation (S2TT)** : Transcription et traduction vocale
- **Text-to-Speech Translation (T2ST)** : Synthèse vocale multilingue
- **Text-to-Text Translation (T2TT)** : Traduction textuelle

## 🚀 Fonctionnalités

### Services disponibles

| Service | Description | Endpoint |
|---------|-------------|----------|
| **S2ST** | Traduction vocale (audio → audio) | `POST /api/v1/s2st` |
| **S2TT** | Transcription vocale (audio → texte) | `POST /api/v1/s2tt` |
| **T2ST** | Synthèse vocale (texte → audio) | `POST /api/v1/t2st` |
| **T2TT** | Traduction textuelle (texte → texte) | `POST /api/v1/t2tt` |

### Langues supportées

L'API supporte **40+ langues** dont :
- Français, Anglais, Espagnol, Allemand, Italien
- Chinois, Japonais, Coréen, Arabe
- Russe, Portugais, Hindi, et bien d'autres

Consultez l'endpoint `/api/v1/languages` pour la liste complète.

## 🛠️ Installation

### Prérequis

- **Python** : 3.8+ (3.10 recommandé)
- **CUDA** : 12.6+ (optimisé pour cette version)
- **Drivers NVIDIA** : 561.17+ (compatible avec votre configuration)
- **RAM** : 16GB+ (32GB+ recommandé pour GPU)
- **VRAM** : 24GB+ (nécessaire pour le modèle large)
- **Espace disque** : 10GB+ (20GB+ recommandé pour les caches)
- **NVIDIA Container Toolkit** : Pour le support Docker GPU

### Installation du NVIDIA Container Toolkit

```bash
# Ajouter le dépôt NVIDIA
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
&& curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
&& curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Installer le toolkit
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Vérifier l'installation
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

### Étapes d'installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-repo/seamlessm4t_api.git
cd seamlessm4t_api

# Créer un environnement virtuel (recommandé)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'API
python main.py
```

## 📡 Utilisation

### Lancement de l'API

```bash
# Lancer uniquement l'API FastAPI
python main.py --api

# Lancer uniquement l'interface Gradio
python main.py --gradio

# Lancer les deux (par défaut)
python main.py --both

# Mode debug
python main.py --debug
```

### Exemples d'utilisation

#### 1. Traduction vocale (S2ST)

```bash
curl -X POST "http://localhost:8000/api/v1/s2st" \
  -F "audio_file=@audio_francais.wav" \
  -F "src_lang=fra" \
  -F "tgt_lang=eng"
```

#### 2. Transcription vocale (S2TT)

```bash
curl -X POST "http://localhost:8000/api/v1/s2tt" \
  -F "audio_file=@audio_francais.wav" \
  -F "src_lang=fra" \
  -F "tgt_lang=eng"
```

#### 3. Synthèse vocale (T2ST)

```bash
curl -X POST "http://localhost:8000/api/v1/t2st" \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour le monde", "src_lang": "fra", "tgt_lang": "eng"}'
```

#### 4. Traduction textuelle (T2TT)

```bash
curl -X POST "http://localhost:8000/api/v1/t2tt" \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour le monde", "src_lang": "fra", "tgt_lang": "eng"}'
```

## 🎯 Interface Gradio

L'API inclut une interface utilisateur Gradio pour tester facilement toutes les fonctionnalités :

```bash
# Accéder à l'interface
http://localhost:7860
```

![Interface Gradio](docs/gradio_interface.png)

## 📊 Performances

### Configuration recommandée

- **GPU** : NVIDIA RTX 3090/4090 ou A100 (avec drivers 561.17+)
- **CUDA** : Version 12.6+ (optimisé pour cette version)
- **CPU** : 8+ cœurs (16+ recommandé pour les charges lourdes)
- **RAM** : 32GB+ (64GB+ recommandé pour les traitements par lots)
- **VRAM** : 24GB+ (nécessaire pour le modèle seamless-m4t-v2-large)

### Optimisations

- **Chargement unique du modèle** : Le modèle est chargé une seule fois et partagé entre tous les services via un singleton
- **Gestion GPU optimisée** : Nettoyage automatique de la mémoire GPU après 5 requêtes
- **Traitement par segments** : Gestion automatique des audios/textes longs (>60s ou >5000 caractères)
- **Support CUDA 12.6** : Optimisations spécifiques pour les dernières architectures GPU
- **Docker optimisé** : Configuration spécialement adaptée pour CUDA 12.6 et drivers 561.17+

### Benchmarks (estimations)

| Configuration | Temps de chargement | Latence moyenne | Débit |
|---------------|-------------------|-----------------|--------|
| RTX 3090 (24GB) | ~15-20s | ~2-5s/requête | ~12 req/min |
| RTX 4090 (24GB) | ~10-15s | ~1-3s/requête | ~20 req/min |
| A100 (40GB) | ~8-12s | ~0.5-2s/requête | ~30 req/min |

## 🔧 Configuration

Modifiez le fichier `config.py` pour personnaliser :

```python
# Ports
FASTAPI_PORT = 8000
GRADIO_SERVER_PORT = 7860

# Modèle
MODEL_NAME = "facebook/seamless-m4t-v2-large"
SAMPLING_RATE = 16000

# Limites
MAX_AUDIO_DURATION = 60  # secondes
MAX_TEXT_LENGTH = 5000   # caractères

# GPU
USE_GPU = True
GPU_CLEANUP_INTERVAL = 5  # Nettoyer après N requêtes
```

## 🧪 Tests

```bash
# Tester l'état de santé
curl http://localhost:8000/api/v1/health

# Lister les langues supportées
curl http://localhost:8000/api/v1/languages

# Accéder à la documentation Swagger
http://localhost:8000/docs

# Accéder à la documentation ReDoc
http://localhost:8000/redoc
```

## 📦 Déploiement

### Avec Docker (recommandé)

```bash
# Construire l'image
docker-compose build

# Lancer les conteneurs
docker-compose up -d

# Vérifier les logs
docker-compose logs -f

# Arrêter les conteneurs
docker-compose down
```

**Configuration requise pour Docker**:
- NVIDIA Container Toolkit installé
- Drivers NVIDIA 561.17+ (recommandé)
- CUDA 12.6+ (optimisé pour cette version)
- Docker 20.10+

**Fonctionnalités Docker**:
- Support GPU complet avec CUDA 12.6
- Environnement Python isolé
- Persistance des fichiers audio et résultats
- Health checks intégrés
- Configuration optimisée pour les performances

### Avec systemd

```ini
# /etc/systemd/system/seamlessm4t-api.service
[Unit]
Description=SeamlessM4T API Service
After=network.target

[Service]
User=your_user
WorkingDirectory=/path/to/seamlessm4t_api
ExecStart=/path/to/venv/bin/python main.py --api
Restart=always
Environment="PATH=/path/to/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

[Install]
WantedBy=multi-user.target
```

## 🤝 Contribution

Les contributions sont les bienvenues ! Veuillez suivre ces étapes :

1. Forker le projet
2. Créer une branche (`git checkout -b feature/ma-fonctionnalite`)
3. Commiter vos changements (`git commit -m 'Ajout de ma fonctionnalité'`)
4. Pusher la branche (`git push origin feature/ma-fonctionnalite`)
5. Ouvrir une Pull Request

## 📜 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 📞 Support

Pour toute question ou problème :
- Ouvrir une issue sur GitHub
- Consulter la [documentation officielle SeamlessM4T](https://github.com/facebookresearch/seamless_communication)

## 🎓 Exemples avancés

### Traitement par lots

```python
import requests
import os

def batch_translate(audio_files, src_lang, tgt_lang):
    results = []
    for audio_file in audio_files:
        with open(audio_file, 'rb') as f:
            files = {'audio_file': f}
            data = {'src_lang': src_lang, 'tgt_lang': tgt_lang}
            
            response = requests.post(
                'http://localhost:8000/api/v1/s2st',
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                results.append(response.json())
    
    return results

# Utilisation
files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
results = batch_translate(files, 'fra', 'eng')
```

### Intégration avec d'autres services

```python
from fastapi import FastAPI
import httpx

app = FastAPI()

SEAMLESS_API_URL = "http://localhost:8000/api/v1"

@app.post("/translate-audio")
async def translate_audio(audio_url: str, src_lang: str, tgt_lang: str):
    async with httpx.AsyncClient() as client:
        # Télécharger l'audio
        audio_response = await client.get(audio_url)
        
        # Envoyer à SeamlessM4T API
        files = {'audio_file': audio_response.content}
        data = {'src_lang': src_lang, 'tgt_lang': tgt_lang}
        
        response = await client.post(
            f"{SEAMLESS_API_URL}/s2st",
            files=files,
            data=data
        )
        
        return response.json()
```

## 🚨 Limitations

- Durée maximale des audios : 60 secondes (découpage automatique pour les audios plus longs)
- Longueur maximale des textes : 5000 caractères (découpage automatique pour les textes plus longs)
- Nécessite une connexion internet pour le téléchargement initial du modèle

## 📊 Métriques

L'API expose des métriques de santé et d'utilisation :

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "gpu_available": true,
  "gpu_memory": {
    "total": 23.7,
    "free": 12.5,
    "used": 11.2
  },
  "services": {
    "s2st": {"model_loaded": true, "device": "cuda", ...},
    "s2tt": {"model_loaded": true, "device": "cuda", ...},
    "t2st": {"model_loaded": true, "device": "cuda", ...},
    "t2tt": {"model_loaded": true, "device": "cuda", ...}
  }
}
```

## 🔍 Dépannage

### Problèmes courants

**1. Erreur de mémoire GPU**
- Solution : Réduire `MAX_AUDIO_DURATION` ou utiliser un GPU avec plus de VRAM

**2. Modèle non chargé**
- Solution : Vérifier la connexion internet et l'espace disque

**3. Erreurs de langue non supportée**
- Solution : Vérifier les codes de langue avec `/api/v1/languages`

### Logs

Les logs sont disponibles dans la console et peuvent être redirigés vers un fichier :

```bash
python main.py > api.log 2>&1 &
```

## 📈 Roadmap

- Ajout de l'authentification JWT
- Support des websockets pour le streaming
- Intégration avec d'autres modèles de traduction
- Optimisation pour les déploiements serverless

## 🙏 Remerciements

- L'équipe Facebook Research pour le modèle SeamlessM4T
- La communauté HuggingFace pour les outils Transformers
- Tous les contributeurs open source

---

© 2024 SeamlessM4T API | Version 1.0.0