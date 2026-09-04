# Guide de déploiement

Deux composants déployés indépendamment, gratuitement — même montage qu'au Projet 8 :

1. **API FastAPI** (`api/`) -> **Render**, service web Docker, plan gratuit. Charge le checkpoint SegFormer MiT-B0 et expose `/predict`.
2. **Dashboard Streamlit** (`dashboard/`) -> **Streamlit Community Cloud**. Consomme l'API et affiche les masques.

Le déploiement s'appuie sur un dépôt **p9-deploy** séparé (uniquement les fichiers nécessaires à l'exécution : ni notebooks, ni données, ni poids d'entraînement B2/U-Net), généré par `scripts/build_deploy_repo.sh`. Le checkpoint `segformer_b0_best.pt` (~14 Mo) y est versionné via **Git LFS**.

URLs en production :

- **Dashboard** : https://p9-segmentation.streamlit.app
- **API** : https://p9-segmentation-api.onrender.com
- **Dépôt de déploiement** : https://github.com/arthur-openclassroom/p9-deploy

> Pourquoi Render et non Hugging Face Spaces comme au Projet 8 : HF a rendu le SDK **Docker payant** et impose ZeroGPU aux Spaces Gradio du tier gratuit, ce qui ne convient pas à une API FastAPI. Render conserve le Dockerfile tel quel. Les 512 Mo du plan gratuit suffisent : l'image mesurée consomme 236 Mio après une inférence en pleine résolution.

---

## 1. Lancer en local

### API

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r api/requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Vérifications :

```bash
curl http://localhost:8000/health
curl http://localhost:8000/categories
```

### Dashboard

Dans un autre terminal :

```bash
pip install -r requirements.txt
export API_URL=http://localhost:8000
streamlit run dashboard/app.py
```

Ouvrir http://localhost:8501.

---

## 2. Générer le dépôt de déploiement

```bash
bash scripts/build_deploy_repo.sh
```

Crée `../p9-deploy/` (~31 Mo, 30 fichiers), initialise git + Git LFS et fait le premier commit. Contenu : `api/`, `dashboard/` (avec `samples/` et `.streamlit/`), `scripts/cityscapes_utils.py`, `models/` (checkpoint B0 + les deux JSON lus par le dashboard), `Dockerfile`, `render.yaml`, `requirements.txt`, `README.md`.

Prérequis : `git-lfs` installé (`brew install git-lfs`).

Puis pousser sur GitHub :

```bash
cd ../p9-deploy
gh repo create arthur-openclassroom/p9-deploy --public --source=. --remote=origin --push
```

---

## 3. Déploiement de l'API sur Render

L'infrastructure est déclarée dans `render.yaml` (service web Docker, plan gratuit, région Francfort, health check sur `/health`).

1. https://dashboard.render.com -> **Blueprints** -> **New Blueprint Instance**.
2. Le workspace n'ayant pas d'intégration GitHub, utiliser le champ **Public Git Repository** : `https://github.com/arthur-openclassroom/p9-deploy`, puis *Continue*.
3. Nommer le blueprint (`p9-segmentation`), branche `main`, chemin `render.yaml` par défaut. **Deploy Blueprint**.
4. Render construit l'image et démarre le service. Une fois "Live" :

```bash
curl https://p9-segmentation-api.onrender.com/health
# {"status":"ok","model":"SegFormer MiT-B0","device":"cpu","weights":"/app/models/segformer_b0_best.pt"}
```

Render injecte lui-même la variable `PORT` au runtime, qui écrase la valeur du Dockerfile : le `CMD` écoute `${PORT}`.

Le dépôt étant relié par URL publique et non par l'intégration GitHub, l'**auto-deploy est inactif** : après un push, relancer manuellement via *Manual Deploy* -> *Deploy latest commit*.

---

## 4. Déploiement du dashboard sur Streamlit Community Cloud

1. https://share.streamlit.io -> **Create app** -> *Deploy a public app from GitHub*.
2. Repo `arthur-openclassroom/p9-deploy`, branche `main`, fichier **`dashboard/app.py`**, sous-domaine `p9-segmentation`.
3. **Advanced settings** : Python **3.12**, et dans *Secrets* :

```toml
API_URL = "https://p9-segmentation-api.onrender.com"
```

4. **Deploy**.

Le dashboard lit les images d'exemple embarquées dans `dashboard/samples/`. Sans le secret, il retombe sur `http://localhost:8000` et reste utilisable en local.

---

## 5. Alternative : build Docker local

```bash
docker build -t p9-segmentation-api .
docker run -d --name p9-api -p 8000:7860 p9-segmentation-api
curl http://localhost:8000/health
```

---

## 6. Variables d'environnement

| Variable | Composant | Description | Défaut |
|----------|-----------|-------------|--------|
| `MODEL_PATH` | api | Chemin du checkpoint `.pt` | `models/segformer_b0_best.pt` (local) / `/app/models/segformer_b0_best.pt` (Docker) |
| `PORT` | api | Port d'écoute, fixé par la plateforme | 7860 (Docker) / 8000 (local) |
| `API_URL` | dashboard | URL publique de l'API | `http://localhost:8000` |

---

## 7. Limites connues

- **Mise en veille** : le plan gratuit Render suspend le service après 15 minutes d'inactivité ; le réveil prend environ 50 secondes. Le dashboard laisse 90 secondes à l'API et affiche un message dédié. Ouvrir le dashboard quelques minutes avant une démonstration.
- L'API renvoie le masque à la **taille de l'image d'entrée** (argmax puis redimensionnement NEAREST côté serveur, pour ne pas allouer de logits pleine résolution).
- Le modèle est reconstruit hors ligne depuis sa configuration (`HF_HUB_OFFLINE=1`) : aucun appel réseau au démarrage du conteneur.

---

## 8. Mesures de validation

| Vérification | Résultat |
|---|---|
| Build Docker local | OK |
| `/health` en production | `{"status":"ok","model":"SegFormer MiT-B0","device":"cpu"}` |
| `/predict` en production | HTTP 200, masque PNG, 1,5 s |
| `/predict/json` en production | 8 catégories, répartition identique au local (flat 37,16 %, construction 34,59 %) |
| Empreinte mémoire du conteneur | 236 Mio après inférence 2048x1024 (plafond gratuit : 512 Mo) |
| Chaîne complète | Upload sur le dashboard déployé -> masque affiché |
