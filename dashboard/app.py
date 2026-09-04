"""
Dashboard Streamlit - Segmentation semantique pour vehicules autonomes.
Future Vision Transport | Arthur Lambotte

Fonctionnalites :
- Exploration des donnees Cityscapes (exemples reels, distribution, transformations)
- Prediction de segmentation via l'API
- Visualisation des resultats
- Accessibilite WCAG
"""

import json
import os
import sys
from pathlib import Path

import streamlit as st
import numpy as np
import requests
import io
from PIL import Image, ImageFilter, ImageOps
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "scripts"))
from cityscapes_utils import (
    CATEGORIES, CATEGORY_COLORS, N_CLASSES,
    map_labels_to_categories, mask_to_rgb,
)

# -- Configuration --
st.set_page_config(
    page_title="Segmentation Semantique - Future Vision Transport",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

def default_api_url():
    """Secret Streamlit Cloud, sinon variable d'environnement, sinon local."""
    try:
        if "API_URL" in st.secrets:
            return st.secrets["API_URL"]
    except Exception:
        # Aucun secrets.toml en local : st.secrets leve, ce n'est pas une erreur.
        pass
    return os.environ.get("API_URL", "http://localhost:8000")


API_URL = st.sidebar.text_input(
    "URL de l'API",
    value=default_api_url(),
    help="Par defaut : secret API_URL (Streamlit Cloud) ou variable d'environnement API_URL",
)

CATEGORY_COLORS_HEX = {
    name: "#{:02x}{:02x}{:02x}".format(*CATEGORY_COLORS[cat_id])
    for cat_id, name in CATEGORIES.items()
}

SAMPLES_DIR = Path(__file__).parent / "samples"
N_SAMPLES = 6


def text_color_for(hex_color):
    """Noir ou blanc selon la luminance du fond, pour un contraste WCAG suffisant."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#000000" if luminance > 140 else "#ffffff"


@st.cache_data
def load_samples():
    """Charge les paires (image, masque de categories) reelles bundlees avec le dashboard."""
    samples = []
    for i in range(1, N_SAMPLES + 1):
        img_path = SAMPLES_DIR / f"sample_{i}_img.png"
        label_path = SAMPLES_DIR / f"sample_{i}_labelids.png"
        if img_path.exists() and label_path.exists():
            img = Image.open(img_path).convert("RGB")
            label = np.array(Image.open(label_path).resize(img.size, resample=Image.NEAREST))
            cat_mask = map_labels_to_categories(label)
            samples.append((img, cat_mask))
    return samples


@st.cache_data
def compute_class_distribution(_samples):
    """Distribution reelle des categories (en pixels) sur les images d'exemple bundlees."""
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for _, cat_mask in _samples:
        for c in range(N_CLASSES):
            counts[c] += np.sum(cat_mask == c)
    total = counts.sum()
    return counts, counts / total * 100


def load_reference_results():
    """Charge les resultats reels (baseline + SegFormer) s'ils sont disponibles.

    Prefere comparison_results.json (notebook 03) pour les DEUX modeles : le U-Net
    y est re-evalue dans l'environnement P9 avec le meme code d'evaluation que le
    SegFormer - meme referentiel. Repli sur les resultats historiques du Projet 8.
    """
    unet_path = ROOT / "models" / "unet_baseline_results.json"
    comparison_path = ROOT / "models" / "comparison_results.json"

    unet_miou, unet_iou_per_class = None, None
    segformer_miou, segformer_iou_per_class = None, None

    if comparison_path.exists():
        data = json.load(open(comparison_path))
        segformer_miou = data["segformer_miou"]
        segformer_iou_per_class = data["segformer_iou_per_class"]
        unet_miou = data.get("unet_miou")
        unet_iou_per_class = data.get("unet_iou_per_class")

    if unet_miou is None and unet_path.exists():
        data = json.load(open(unet_path))["unet_scratch"]
        unet_miou = data["val_iou_mean"]
        unet_iou_per_class = data["iou_per_class"]

    return unet_miou, unet_iou_per_class, segformer_miou, segformer_iou_per_class


# -- Sidebar navigation --
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Section",
    ["Accueil", "Exploration des donnees", "Prediction", "A propos"],
    label_visibility="collapsed",
)

(UNET_MIOU, UNET_IOU_PER_CLASS,
 SEGFORMER_MIOU, SEGFORMER_IOU_PER_CLASS) = load_reference_results()

# ============================================================
# ACCUEIL
# ============================================================
if page == "Accueil":
    st.title("Segmentation semantique d'images pour vehicules autonomes")
    st.markdown("**Future Vision Transport** - Equipe R&D")

    st.markdown("""
    ### Contexte

    Ce dashboard presente les resultats du projet de segmentation semantique
    d'images de scenes urbaines pour le systeme embarque de vision par ordinateur
    des vehicules autonomes.

    ### Modeles
    """)

    unet_txt = f"{UNET_MIOU:.3f}" if UNET_MIOU is not None else "non disponible"
    segformer_txt = f"{SEGFORMER_MIOU:.3f}" if SEGFORMER_MIOU is not None else "non entraine"

    st.markdown(f"""
    | Modele | Architecture | mIoU (validation) |
    |--------|-------------|------|
    | Baseline (Projet 8) | U-Net from scratch | {unet_txt} |
    | **Nouvel algorithme** | **SegFormer MiT-B0 (Transformer)** | **{segformer_txt}** |

    ### Categories de segmentation

    Le modele segmente chaque pixel en **8 categories principales** :
    """)

    cols = st.columns(4)
    for i, (cat_id, cat_name) in enumerate(CATEGORIES.items()):
        col = cols[i % 4]
        color = CATEGORY_COLORS_HEX[cat_name]
        col.markdown(
            f'<div style="background-color:{color};color:{text_color_for(color)};padding:10px;'
            f'border-radius:5px;margin:5px;text-align:center;font-weight:bold;">'
            f'{cat_name}</div>',
            unsafe_allow_html=True,
        )

# ============================================================
# EXPLORATION DES DONNEES
# ============================================================
elif page == "Exploration des donnees":
    st.title("Exploration du dataset Cityscapes")

    st.markdown("""
    Le dataset **Cityscapes** contient 5000 images annotees de scenes urbaines,
    capturees dans 50 villes europeennes. Chaque image est accompagnee d'un
    masque de segmentation pixel-level.
    """)

    st.subheader("Statistiques du dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric("Images (train)", "2 975")
    col2.metric("Images (val)", "500")
    col3.metric("Resolution native", "2048 x 1024")

    samples = load_samples()

    st.subheader("Exemples d'images du dataset")
    if samples:
        st.markdown(f"{len(samples)} images reelles du set de validation, avec leur masque de segmentation.")
        cols = st.columns(3)
        for i, (img, cat_mask) in enumerate(samples):
            col = cols[i % 3]
            overlay = Image.blend(img.convert("RGB"), Image.fromarray(mask_to_rgb(cat_mask)).resize(img.size), alpha=0.5)
            col.image(img, caption=f"Image {i + 1}", use_container_width=True)
            col.image(overlay, caption=f"Superposition masque {i + 1}", use_container_width=True)
    else:
        st.warning("Aucune image d'exemple trouvee dans dashboard/samples/.")

    st.subheader("Exemple de transformation d'image")
    st.markdown(
        "Illustration des transformations utilisees dans le pipeline d'augmentation "
        "(floutage gaussien, egalisation d'histogramme)."
    )
    if samples:
        base_img = samples[0][0]
        blurred = base_img.filter(ImageFilter.GaussianBlur(radius=4))
        equalized = ImageOps.equalize(base_img)
        cols = st.columns(3)
        cols[0].image(base_img, caption="Image originale", use_container_width=True)
        cols[1].image(blurred, caption="Floutage gaussien", use_container_width=True)
        cols[2].image(equalized, caption="Egalisation d'histogramme", use_container_width=True)

    st.subheader("Distribution des categories")
    if samples:
        counts, proportions = compute_class_distribution(samples)
        st.caption(f"Comptage de pixels reel sur les {len(samples)} images d'exemple bundlees.")
        dist_data = {
            "Categorie": [CATEGORIES[i] for i in range(N_CLASSES)],
            "Proportion (%)": proportions.round(2),
        }

        fig_bar = px.bar(
            dist_data,
            x="Categorie",
            y="Proportion (%)",
            color="Categorie",
            color_discrete_map=CATEGORY_COLORS_HEX,
            title="Distribution des pixels par categorie",
        )
        fig_bar.update_layout(showlegend=False, font=dict(size=14), title_font_size=16)
        fig_bar.update_traces(marker_line_color="black", marker_line_width=1.5)
        st.plotly_chart(fig_bar, use_container_width=True)

        fig_pie = px.pie(
            dist_data,
            values="Proportion (%)",
            names="Categorie",
            title="Repartition des categories",
            color="Categorie",
            color_discrete_map=CATEGORY_COLORS_HEX,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label", textfont_size=12)
        fig_pie.update_layout(font=dict(size=14), title_font_size=16)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Comparaison des modeles")
    if UNET_IOU_PER_CLASS is not None:
        cat_names = list(CATEGORIES.values())
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="U-Net (baseline)",
            x=cat_names,
            y=[UNET_IOU_PER_CLASS[c] for c in cat_names],
            marker_color="steelblue",
            marker_line_color="black",
            marker_line_width=1,
        ))
        if SEGFORMER_IOU_PER_CLASS is not None:
            fig_comp.add_trace(go.Bar(
                name="SegFormer (MiT-B0)",
                x=cat_names,
                y=[SEGFORMER_IOU_PER_CLASS[c] for c in cat_names],
                marker_color="coral",
                marker_line_color="black",
                marker_line_width=1,
            ))
        fig_comp.update_layout(
            title="IoU par categorie : U-Net vs SegFormer",
            yaxis_title="IoU",
            barmode="group",
            font=dict(size=14),
            title_font_size=16,
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        if SEGFORMER_IOU_PER_CLASS is None:
            st.info("SegFormer pas encore entraine : executez le notebook 03 pour completer la comparaison.")
    else:
        st.warning("Resultats de la baseline non trouves (models/unet_baseline_results.json).")

# ============================================================
# PREDICTION
# ============================================================
elif page == "Prediction":
    st.title("Prediction de segmentation")
    st.markdown(
        "Telechargez une image de scene urbaine pour obtenir "
        "sa segmentation semantique via le modele SegFormer."
    )

    uploaded_file = st.file_uploader(
        "Choisir une image",
        type=["png", "jpg", "jpeg"],
        help="Image de scene urbaine (format PNG ou JPEG)",
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Image originale")
            st.image(img, use_container_width=True)

        # Appel a l'API
        with st.spinner("Segmentation en cours..."):
            try:
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                # Timeout large : le tier gratuit (Render) peut mettre ~1 min a sortir de veille
                response = requests.post(f"{API_URL}/predict", files=files, timeout=90)

                if response.status_code == 200:
                    mask_img = Image.open(io.BytesIO(response.content))
                    with col2:
                        st.subheader("Masque de segmentation")
                        st.image(mask_img, use_container_width=True)

                    # Superposition
                    st.subheader("Superposition image + masque")
                    img_resized = img.resize(mask_img.size)
                    overlay = Image.blend(img_resized, mask_img, alpha=0.5)
                    st.image(overlay, use_container_width=True)

                    # Distribution JSON
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    json_resp = requests.post(
                        f"{API_URL}/predict/json", files=files, timeout=90
                    )
                    if json_resp.status_code == 200:
                        data = json_resp.json()
                        st.subheader("Distribution des categories")

                        dist = data.get("distribution", {})
                        if dist:
                            cats = list(dist.keys())
                            props = [dist[c]["proportion"] for c in cats]
                            fig = px.bar(
                                x=cats, y=props,
                                labels={"x": "Categorie", "y": "Proportion (%)"},
                                color=cats,
                                color_discrete_map=CATEGORY_COLORS_HEX,
                                title="Categories detectees dans l'image",
                            )
                            fig.update_traces(
                                marker_line_color="black",
                                marker_line_width=1.5,
                            )
                            fig.update_layout(
                                showlegend=False,
                                font=dict(size=14),
                            )
                            st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error(
                        f"Erreur de l'API (code {response.status_code}). "
                        f"Verifiez que l'API est lancee sur {API_URL}"
                    )

            except requests.exceptions.ConnectionError:
                st.warning(
                    f"Impossible de se connecter a l'API ({API_URL}). "
                    "Lancez l'API avec : `uvicorn api.main:app --reload`"
                )
            except requests.exceptions.Timeout:
                st.warning(
                    f"L'API ({API_URL}) n'a pas repondu a temps. Si elle est hebergee "
                    "sur un tier gratuit, elle sort peut-etre de veille : reessayez dans une minute."
                )
            except requests.exceptions.RequestException as exc:
                st.error(f"Erreur lors de l'appel a l'API : {exc}")

    # Legende des categories
    st.subheader("Legende des categories")
    cols = st.columns(4)
    for i, (cat_id, cat_name) in enumerate(CATEGORIES.items()):
        col = cols[i % 4]
        color = CATEGORY_COLORS_HEX[cat_name]
        col.markdown(
            f'<div style="background-color:{color};color:{text_color_for(color)};padding:8px;'
            f'border-radius:4px;margin:3px;text-align:center;">'
            f'{cat_name}</div>',
            unsafe_allow_html=True,
        )

# ============================================================
# A PROPOS
# ============================================================
elif page == "A propos":
    st.title("A propos")
    st.markdown("""
    ### Projet 9 - Segmentation semantique pour vehicules autonomes

    **Entreprise** : Future Vision Transport
    **Equipe** : R&D - Bloc segmentation d'images
    **Auteur** : Arthur Lambotte

    ### Baseline

    **U-Net from scratch** (~31M parametres), reutilise du Projet 8 : encodeur/decodeur
    convolutif entraine sur les 2 975 images d'entrainement de Cityscapes.

    ### Nouvel algorithme

    **SegFormer** (Xie et al., NeurIPS 2021), variante **MiT-B0**, est un modele de
    segmentation semantique base sur les Vision Transformers. Il combine un encodeur
    hierarchique (Mix Transformer - MiT), pre-entraine sur ImageNet, et un decodeur
    MLP leger, reentraine sur les 8 categories Cityscapes.

    ### References

    1. Xie, E. et al. (2021). *SegFormer: Simple and Efficient Design for
       Semantic Segmentation with Transformers.* NeurIPS 2021.
    2. Ronneberger, O. et al. (2015). *U-Net: Convolutional Networks for
       Biomedical Image Segmentation.* MICCAI 2015.
    3. Cordts, M. et al. (2016). *The Cityscapes Dataset for Semantic Urban
       Scene Understanding.* CVPR 2016.

    ### Accessibilite

    Ce dashboard a ete concu en suivant les recommandations WCAG 2.1 :
    - Contrastes de couleurs suffisants (texte noir ou blanc choisi selon la luminance du fond)
    - Textes lisibles et redimensionnables
    - Navigation au clavier possible
    - Labels descriptifs sur tous les elements interactifs
    - Graphiques avec bordures pour distinguer les categories sans la couleur
    """)
