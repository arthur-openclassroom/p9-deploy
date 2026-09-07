# P9 — API de segmentation sémantique + dashboard

Dépôt de **déploiement** de la preuve de concept du Projet 9 : il ne contient
que les fichiers nécessaires à l'exécution. Le projet complet (notebooks,
documents, résultats) vit dans le dépôt principal.

**Démo en ligne**
- Dashboard (Streamlit Community Cloud) : https://p9-segmentation.streamlit.app
- API (Render, service web Docker) : https://p9-segmentation-api.onrender.com

L'API est hébergée sur le plan gratuit de Render : elle se met en veille après
15 minutes d'inactivité et met environ 50 secondes à se réveiller.

## Composants

- **API** : FastAPI + SegFormer MiT-B0 fine-tuné sur Cityscapes (8 catégories),
  servi en Docker. Le modèle est reconstruit hors ligne depuis sa configuration,
  sans appel réseau au démarrage.
- **Dashboard** : Streamlit, consomme l'API via le secret `API_URL`.

## Endpoints

| Méthode | Route | Description |
|---|---|---|
| GET | `/health` | état de l'API, device, chemin des poids |
| GET | `/categories` | les 8 catégories et leurs couleurs |
| POST | `/predict` | image -> masque de segmentation en PNG |
| POST | `/predict/json` | image -> répartition des catégories en JSON |

## Prérequis : Git LFS

Le checkpoint `models/segformer_b0_best.pt` (14 Mo) est versionné avec **Git LFS**.
Sans `git-lfs` installé **avant** le clone, git récupère un pointeur de 133 octets
et l'API échoue au chargement des poids.

```bash
brew install git-lfs   # ou apt install git-lfs
git lfs install
git clone https://github.com/arthur-openclassroom/p9-deploy.git
```

Si le dépôt est déjà cloné sans LFS : `git lfs install && git lfs pull`.

## Lancer en local

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000

# dans un autre terminal
pip install -r requirements.txt
API_URL=http://localhost:8000 streamlit run dashboard/app.py
```

## Variables d'environnement

| Variable | Composant | Description | Défaut |
|---|---|---|---|
| `MODEL_PATH` | api | chemin du checkpoint `.pt` | `models/segformer_b0_best.pt` |
| `PORT` | api | port d'écoute, fixé par la plateforme | 7860 (Docker) / 8000 (local) |
| `API_URL` | dashboard | URL publique de l'API | `http://localhost:8000` |

Le déploiement est déclaré dans `render.yaml`. Procédure complète : `DEPLOIEMENT.md`
du dépôt principal.
