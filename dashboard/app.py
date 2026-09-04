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
CLASSES_PAR_COULEUR = {tuple(c): i for i, c in CATEGORY_COLORS.items()}


def rgb_vers_classes(mask_rgb):
    """Masque RGB renvoye par l'API -> carte de classes (H, W) uint8."""
    arr = np.asarray(mask_rgb.convert("RGB"))
    classes = np.zeros(arr.shape[:2], dtype=np.uint8)
    for cat_id, color in CATEGORY_COLORS.items():
        classes[np.all(arr == np.array(color), axis=-1)] = cat_id
    return classes


def miou_image(pred, gt):
    """mIoU sur une seule image, moyenne sur les categories presentes."""
    ious = []
    for c in range(N_CLASSES):
        union = np.logical_or(pred == c, gt == c).sum()
        if union:
            ious.append(np.logical_and(pred == c, gt == c).sum() / union)
    return float(np.mean(ious)) if ious else 0.0


def carte_ecarts(a, b, couleur=(214, 39, 40)):
    """Pixels ou a et b different, en rouge sur fond clair."""
    ecart = a != b
    out = np.full((*ecart.shape, 3), 240, dtype=np.uint8)
    out[ecart] = couleur
    return out, float(ecart.mean() * 100)


def masque_segformer(nom, contenu, mime):
    """Appelle l'API et renvoie le masque RGB, ou leve l'exception requests."""
    return Image.open(
        io.BytesIO(
            requests.post(
                f"{API_URL}/predict",
                files={"file": (nom, io.BytesIO(contenu), mime)},
                timeout=90,
            ).content
        )
    )


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

    source = st.radio(
        "Source de l'image",
        ("Exemple du dataset", "Importer une image"),
        horizontal=True,
        help="Les exemples du dataset ont une verite terrain : ils permettent de "
             "comparer SegFormer a la baseline U-Net.",
    )

    # ------------------------------------------------------------------
    # A. Exemple du dataset : comparaison complete
    # ------------------------------------------------------------------
    if source == "Exemple du dataset":
        st.markdown(
            "Chaque exemple est fourni avec son **annotation de reference**. "
            "Les predictions du **U-Net** (baseline du Projet 8, 31 M parametres) "
            "sont precalculees : le modele pese 119 Mo et ne peut pas etre servi "
            "en ligne a cote de SegFormer. Celles de **SegFormer** sont calculees "
            "en direct par l'API."
        )

        numero = st.selectbox(
            "Choisir une scene urbaine",
            options=list(range(1, N_SAMPLES + 1)),
            format_func=lambda n: f"Scene {n}",
        )

        img_path = SAMPLES_DIR / f"sample_{numero}_img.png"
        gt_path = SAMPLES_DIR / f"sample_{numero}_gt.png"
        unet_path = SAMPLES_DIR / f"sample_{numero}_unet.png"

        img = Image.open(img_path).convert("RGB")
        gt = np.asarray(Image.open(gt_path))
        unet = np.asarray(Image.open(unet_path))

        segformer = None
        with st.spinner("Segmentation SegFormer en cours..."):
            try:
                masque = masque_segformer(
                    img_path.name, img_path.read_bytes(), "image/png"
                )
                segformer = rgb_vers_classes(masque)
            except requests.exceptions.Timeout:
                st.warning(
                    f"L'API ({API_URL}) n'a pas repondu a temps. Hebergee sur une "
                    "offre gratuite, elle sort peut-etre de veille : reessayez dans "
                    "une minute. La verite terrain et le U-Net restent affiches."
                )
            except requests.exceptions.RequestException as exc:
                st.warning(f"API injoignable ({exc}). Comparaison partielle.")

        colonnes = st.columns(4)
        colonnes[0].markdown("**Image**")
        colonnes[0].image(img, use_container_width=True)
        colonnes[1].markdown("**Verite terrain**")
        colonnes[1].image(mask_to_rgb(gt), use_container_width=True)
        colonnes[2].markdown("**U-Net** (baseline)")
        colonnes[2].image(mask_to_rgb(unet), use_container_width=True)
        colonnes[3].markdown("**SegFormer MiT-B0**")
        if segformer is not None:
            colonnes[3].image(mask_to_rgb(segformer), use_container_width=True)
        else:
            colonnes[3].info("Indisponible")

        st.subheader("Ou les deux modeles se trompent")

        err_unet, taux_unet = carte_ecarts(unet, gt)
        cols_err = st.columns(3)
        cols_err[0].markdown("**Erreurs du U-Net**")
        cols_err[0].image(err_unet, use_container_width=True)
        cols_err[0].caption(f"{taux_unet:.1f} % des pixels mal classes")

        if segformer is not None:
            err_seg, taux_seg = carte_ecarts(segformer, gt)
            cols_err[1].markdown("**Erreurs de SegFormer**")
            cols_err[1].image(err_seg, use_container_width=True)
            cols_err[1].caption(f"{taux_seg:.1f} % des pixels mal classes")

            desaccord, taux_desaccord = carte_ecarts(
                unet, segformer, couleur=(148, 103, 189)
            )
            cols_err[2].markdown("**Desaccord entre les deux modeles**")
            cols_err[2].image(desaccord, use_container_width=True)
            cols_err[2].caption(f"{taux_desaccord:.1f} % des pixels")

            st.subheader("Scores sur cette image")
            m_unet, m_seg = miou_image(unet, gt), miou_image(segformer, gt)
            mesures = st.columns(3)
            mesures[0].metric("mIoU U-Net", f"{m_unet:.3f}")
            mesures[1].metric(
                "mIoU SegFormer MiT-B0",
                f"{m_seg:.3f}",
                delta=f"{m_seg - m_unet:+.3f}",
            )
            mesures[2].metric("Pixels en desaccord", f"{taux_desaccord:.1f} %")

            st.caption(
                "Sur ces images propres, le U-Net garde l'avantage : c'est le "
                "resultat attendu, et il est conforme au mIoU global (0,754 contre "
                "0,674). L'interet de SegFormer se joue ailleurs : sous corruptions "
                "(bruit, flou, brouillard, obscurite), il conserve 78,8 % de sa "
                "performance contre 48,7 % pour le U-Net, pour 12 % des parametres "
                "et une inference 17 fois plus rapide sur CPU."
            )

            ious_unet = {
                CATEGORIES[c]: np.logical_and(unet == c, gt == c).sum()
                / max(np.logical_or(unet == c, gt == c).sum(), 1)
                for c in range(N_CLASSES)
                if np.logical_or(unet == c, gt == c).sum()
            }
            ious_seg = {
                CATEGORIES[c]: np.logical_and(segformer == c, gt == c).sum()
                / max(np.logical_or(segformer == c, gt == c).sum(), 1)
                for c in range(N_CLASSES)
                if np.logical_or(segformer == c, gt == c).sum()
            }
            categories_communes = [c for c in CATEGORIES.values() if c in ious_unet or c in ious_seg]
            figure = go.Figure()
            figure.add_trace(go.Bar(
                name="U-Net (baseline)",
                x=categories_communes,
                y=[ious_unet.get(c, 0) for c in categories_communes],
                marker_color="#7f7f7f",
                marker_pattern_shape="/",
            ))
            figure.add_trace(go.Bar(
                name="SegFormer MiT-B0",
                x=categories_communes,
                y=[ious_seg.get(c, 0) for c in categories_communes],
                marker_color="#4f46e5",
            ))
            figure.update_layout(
                barmode="group",
                title="IoU par categorie, sur cette image",
                yaxis_title="IoU",
                font=dict(size=14),
            )
            st.plotly_chart(figure, use_container_width=True)

    # ------------------------------------------------------------------
    # B. Image importee : SegFormer seul, sans verite terrain
    # ------------------------------------------------------------------
    else:
        st.markdown(
            "Importez une image de scene urbaine pour obtenir sa segmentation "
            "par SegFormer. Sans annotation de reference, la comparaison avec "
            "le U-Net n'est pas possible : utilisez les exemples du dataset "
            "pour cela."
        )

        fichier = st.file_uploader(
            "Choisir une image",
            type=["png", "jpg", "jpeg"],
            help="Image de scene urbaine (format PNG ou JPEG)",
        )

        if fichier is not None:
            img = Image.open(fichier).convert("RGB")
            contenu = fichier.getvalue()
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Image originale")
                st.image(img, use_container_width=True)

            with st.spinner("Segmentation en cours..."):
                try:
                    masque = masque_segformer(fichier.name, contenu, fichier.type)
                    with col2:
                        st.subheader("Masque de segmentation")
                        st.image(masque, use_container_width=True)

                    st.subheader("Superposition image + masque")
                    st.image(
                        Image.blend(img.resize(masque.size), masque.convert("RGB"), alpha=0.5),
                        use_container_width=True,
                    )

                    reponse = requests.post(
                        f"{API_URL}/predict/json",
                        files={"file": (fichier.name, io.BytesIO(contenu), fichier.type)},
                        timeout=90,
                    )
                    if reponse.status_code == 200:
                        distribution = reponse.json().get("distribution", {})
                        if distribution:
                            st.subheader("Distribution des categories")
                            noms = list(distribution.keys())
                            fig = px.bar(
                                x=noms,
                                y=[distribution[c]["proportion"] for c in noms],
                                labels={"x": "Categorie", "y": "Proportion (%)"},
                                color=noms,
                                color_discrete_map=CATEGORY_COLORS_HEX,
                                title="Categories detectees dans l'image",
                            )
                            fig.update_traces(marker_line_color="black", marker_line_width=1.5)
                            fig.update_layout(showlegend=False, font=dict(size=14))
                            st.plotly_chart(fig, use_container_width=True)

                except requests.exceptions.Timeout:
                    st.warning(
                        f"L'API ({API_URL}) n'a pas repondu a temps. Hebergee sur une "
                        "offre gratuite, elle sort peut-etre de veille : reessayez "
                        "dans une minute."
                    )
                except requests.exceptions.ConnectionError:
                    st.warning(
                        f"Impossible de se connecter a l'API ({API_URL}). "
                        "En local : `uvicorn api.main:app --reload`"
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
