"""
Utilitaires pour le dataset Cityscapes.
Mapping des 8 categories principales et fonctions de preprocessing.
"""

import numpy as np
from pathlib import Path

CITYSCAPES_ROOT = Path(__file__).parent.parent / "data" / "cityscapes"

# Les 8 categories principales de Cityscapes
# Mapping : label_id original -> categorie principale (0-7)
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

N_CLASSES = 8

# Couleurs pour la visualisation (une par categorie)
CATEGORY_COLORS = {
    0: (0, 0, 0),        # void - noir
    1: (128, 64, 128),    # flat - violet
    2: (70, 70, 70),      # construction - gris
    3: (153, 153, 153),   # object - gris clair
    4: (107, 142, 35),    # nature - vert
    5: (70, 130, 180),    # sky - bleu
    6: (220, 20, 60),     # human - rouge
    7: (0, 0, 142),       # vehicle - bleu fonce
}

# Mapping des label IDs originaux Cityscapes vers les 8 categories
# Source : https://github.com/mcordts/cityscapesScripts
LABEL_TO_CATEGORY = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,    # void
    7: 1, 8: 1, 9: 1, 10: 1,                         # flat
    11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2,       # construction
    17: 3, 18: 3, 19: 3, 20: 3,                       # object
    21: 4, 22: 4,                                      # nature
    23: 5,                                              # sky
    24: 6, 25: 6,                                      # human
    26: 7, 27: 7, 28: 7, 29: 7, 30: 7, 31: 7, 32: 7, 33: 7,  # vehicle
    -1: 0,                                              # licence plate -> void
}


def map_labels_to_categories(label_img):
    """Convertit une image de labels Cityscapes en 8 categories principales."""
    cat_img = np.zeros_like(label_img, dtype=np.uint8)
    for label_id, cat_id in LABEL_TO_CATEGORY.items():
        cat_img[label_img == label_id] = cat_id
    return cat_img


def mask_to_rgb(mask):
    """Convertit un mask de categories en image RGB pour la visualisation."""
    h, w = mask.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cat_id, color in CATEGORY_COLORS.items():
        rgb[mask == cat_id] = color
    return rgb


def get_image_label_pairs(split="train"):
    """Retourne les paires (image_path, label_path) pour un split donne."""
    img_dir = CITYSCAPES_ROOT / "leftImg8bit" / split
    label_dir = CITYSCAPES_ROOT / "gtFine" / split

    pairs = []
    for img_path in sorted(img_dir.rglob("*_leftImg8bit.png")):
        city = img_path.parent.name
        basename = img_path.name.replace("_leftImg8bit.png", "")
        label_path = label_dir / city / f"{basename}_gtFine_labelIds.png"
        if label_path.exists():
            pairs.append((str(img_path), str(label_path)))

    return pairs


def compute_class_weights(label_paths, n_samples=100):
    """Calcule les poids de classes pour gerer le desequilibre."""
    from PIL import Image

    counts = np.zeros(N_CLASSES, dtype=np.int64)
    sample_paths = label_paths[:n_samples]

    for lp in sample_paths:
        label = np.array(Image.open(lp))
        cat = map_labels_to_categories(label)
        for c in range(N_CLASSES):
            counts[c] += np.sum(cat == c)

    total = counts.sum()
    weights = total / (N_CLASSES * counts + 1e-6)
    weights = weights / weights.sum() * N_CLASSES
    return weights
