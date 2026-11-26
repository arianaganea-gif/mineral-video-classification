##############################################################################################################

# © 2026, Ariana Ganea
# Aquesta obra està subjecta a una llicència de Creative Commons Reconeixement-NoComercial-SenseObraDerivada
# 4.0 Internacional (CC BY-NC-ND 4.0). Veure : http://creativecommons.org/licenses/by-nc-nd/4.0/

##############################################################################################################

# CODI PER MIRAR COM HA QUEDAT LA SEGMENTACIÓ D'ALGUNS MINERALS AMB SAM2

##############################################################################################################

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

BASE = "../Dataset/segmented_minerals"
biotite_dir = os.path.join(BASE, "Biotite")
hornblende_dir = os.path.join(BASE, "Hornblende")

# Fem un crop per engrandir el mineral
def crop_white(img, thr=250):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray < thr
    coords = np.column_stack(np.where(mask))
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = img[y0:y1, x0:x1]

    return cropped


# Funcio per veure el primer frame en PPL
def get_first_frame(sample_path):

    ppl_path = os.path.join(sample_path, "PPL_frames")
    if not os.path.exists(ppl_path):
        return None

    files = sorted(os.listdir(ppl_path))
    img_path = os.path.join(ppl_path, files[0])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_white(img)

    return img


def get_25_images(folder):

    samples = sorted(os.listdir(folder))[:25]
    images = []
    for s in samples:
        sample_path = os.path.join(folder, s)
        img = get_first_frame(sample_path)
        if img is not None:
            images.append(img)

    return images


biotite_imgs = get_25_images(biotite_dir)
horn_imgs = get_25_images(hornblende_dir)

fig, axes = plt.subplots(10, 5, figsize=(15, 15))

# Fem les grafiques per la biotita

for i in range(25):

    r = i // 5
    c = i % 5

    if i < len(biotite_imgs):
        axes[r, c].imshow(biotite_imgs[i])

    axes[r, c].axis("off")

axes[0, 2].set_title("Biotita", fontsize=12)

# Grafiquem les hornblendes

for i in range(25):

    r = 5 + (i // 5)
    c = i % 5

    if i < len(horn_imgs):
        axes[r, c].imshow(horn_imgs[i])

    axes[r, c].axis("off")

axes[5, 2].set_title("Hornblenda", fontsize=12)


plt.tight_layout()
plt.show()