---
title: P9 Segmentation API
emoji: 🚗
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# P9 — API de segmentation semantique (SegFormer MiT-B0)

Depot de **deploiement** de la preuve de concept du Projet 9 : il ne contient
que les fichiers necessaires a l'execution. Le projet complet (notebooks,
documents, resultats) vit dans le depot principal.

- **API** : FastAPI + SegFormer MiT-B0 fine-tune sur Cityscapes (8 categories),
  servie en Docker sur Hugging Face Spaces (port 7860).
- **Dashboard** : Streamlit, deploye sur Streamlit Community Cloud, consomme
  l'API via le secret `API_URL`.

## Endpoints

| Methode | Route | Description |
|---|---|---|
| GET | `/health` | etat de l'API, device, chemin des poids |
| GET | `/categories` | les 8 categories et leurs couleurs |
| POST | `/predict` | image -> masque de segmentation en PNG |
| POST | `/predict/json` | image -> repartition des categories en JSON |

## Lancer en local

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000

# dans un autre terminal
pip install -r requirements.txt
API_URL=http://localhost:8000 streamlit run dashboard/app.py
```

## Variables d'environnement

| Variable | Composant | Description | Defaut |
|---|---|---|---|
| `MODEL_PATH` | api | chemin du checkpoint `.pt` | `models/segformer_b0_best.pt` |
| `PORT` | api | port d'ecoute (fixe par la plateforme) | 7860 (Docker) / 8000 (local) |
| `API_URL` | dashboard | URL publique de l'API | `http://localhost:8000` |

Le modele est reconstruit hors ligne depuis sa configuration (`HF_HUB_OFFLINE=1`) :
aucun appel reseau au demarrage du conteneur.
