"""
Chargement du modele SegFormer et fonctions de prediction.
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    SegformerConfig,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

CATEGORIES = {
    0: "void",
    1: "flat",
    2: "construction",
    3: "object",
    4: "nature",
    5: "sky",
    6: "human",
    7: "vehicle",
}

CATEGORY_COLORS = {
    0: (0, 0, 0),
    1: (128, 64, 128),
    2: (70, 70, 70),
    3: (153, 153, 153),
    4: (107, 142, 35),
    5: (70, 130, 180),
    6: (220, 20, 60),
    7: (0, 0, 142),
}

N_CLASSES = 8
MODEL_DIR = Path(__file__).parent.parent / "models"
# MODEL_PATH permet de pointer les poids ailleurs sans toucher au code
# (Docker / Hugging Face Spaces : /app/models/segformer_b0_best.pt).
WEIGHTS_PATH = Path(os.environ.get("MODEL_PATH", MODEL_DIR / "segformer_b0_best.pt"))

# Hyperparametres de nvidia/mit-b0, ecrits en dur pour construire le modele
# SANS appel reseau au demarrage : le checkpoint fine-tune contient deja tous
# les poids, telecharger le backbone ImageNet ne servait qu'a etre ecrase.
MIT_B0 = dict(
    num_channels=3,
    num_encoder_blocks=4,
    depths=[2, 2, 2, 2],
    sr_ratios=[8, 4, 2, 1],
    hidden_sizes=[32, 64, 160, 256],
    patch_sizes=[7, 3, 3, 3],
    strides=[4, 2, 2, 2],
    num_attention_heads=[1, 2, 5, 8],
    mlp_ratios=[4, 4, 4, 4],
    decoder_hidden_size=256,
)


def load_model():
    """Charge le modele SegFormer et le processor."""
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    if device.type == "cpu":
        # Instance cloud a faible nombre de vCPU : le pool par defaut de torch
        # coute de la RAM et n'accelere pas une inference batch 1.
        torch.set_num_threads(1)

    id2label = {i: name for i, name in CATEGORIES.items()}
    label2id = {name: i for i, name in CATEGORIES.items()}

    config = SegformerConfig(
        num_labels=N_CLASSES,
        id2label=id2label,
        label2id=label2id,
        **MIT_B0,
    )
    model = SegformerForSemanticSegmentation(config)

    # Sans les poids fine-tunes, la tete 8 classes est aleatoire : l'API
    # servirait du bruit avec un /health OK. On echoue fort et tot.
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Poids introuvables : {WEIGHTS_PATH} - definir MODEL_PATH ou "
            "executer notebooks/03_segformer.ipynb d'abord."
        )
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state)
    del state
    print(f"Poids charges depuis {WEIGHTS_PATH}")

    model = model.to(device)
    model.eval()

    processor = SegformerImageProcessor(
        do_resize=True,
        size={"height": 256, "width": 256},
        do_normalize=True,
    )

    print(f"Modele charge sur {device}")
    return model, processor, device


def mask_to_rgb(mask):
    """Convertit un masque de categories en image RGB."""
    h, w = mask.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cat_id, color in CATEGORY_COLORS.items():
        rgb[mask == cat_id] = color
    return rgb


@torch.no_grad()
def predict(model, processor, device, img_array, return_classes=False):
    """
    Segmente une image.

    Args:
        img_array: image numpy (H, W, 3) en uint8
        return_classes: si True, renvoie le masque de classes au lieu du RGB

    Returns:
        masque RGB (H, W, 3) ou masque de classes (H, W)
    """
    encoded = processor(images=img_array, return_tensors="pt")
    pixel_values = encoded["pixel_values"].to(device)

    outputs = model(pixel_values=pixel_values)
    # argmax AVANT le reechantillonnage : interpoler les logits vers la taille
    # native allouerait un tenseur (1, 8, H, W) float32, soit 67 Mo pour une
    # image Cityscapes 2048x1024. Le masque uint8 remis a l'echelle en NEAREST
    # donne exactement le meme resultat pour 2 Mo.
    small = outputs.logits.argmax(dim=1).squeeze().to(torch.uint8).cpu().numpy()
    del outputs

    h, w = img_array.shape[0], img_array.shape[1]
    pred_classes = np.array(
        Image.fromarray(small).resize((w, h), resample=Image.NEAREST)
    )

    if return_classes:
        return pred_classes
    return mask_to_rgb(pred_classes)
