##############################################################################################################

# © 2026, Ariana Ganea
# Aquesta obra està subjecta a una llicència de Creative Commons Reconeixement-NoComercial-SenseObraDerivada
# 4.0 Internacional (CC BY-NC-ND 4.0). Veure : http://creativecommons.org/licenses/by-nc-nd/4.0/

##############################################################################################################

# CODI per transformar els videos en tensors per tenir el dataset més compactat, afegint una dimensió més
# que és PPL,XPL

##############################################################################################################

import os
import torch
from torchvision import transforms
from PIL import Image

DATASET_PATH = "../Dataset/segmented_minerals"
OUTPUT_PATH = "../Dataset/segmented_minerals_tensor"
MAX_FRAMES = 72

os.makedirs(OUTPUT_PATH, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((280,280)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

classes = {"Biotite":0,"Hornblende":1}

for mineral in classes:

    mineral_path = os.path.join(DATASET_PATH, mineral)

    for sample in os.listdir(mineral_path):

        sample_path = os.path.join(mineral_path, sample)

        ppl_folder = os.path.join(sample_path,"PPL_frames")
        xpl_folder = os.path.join(sample_path,"XPL_frames")

        if not os.path.exists(ppl_folder):
            continue

        frames = sorted(
            os.listdir(ppl_folder),
            key=lambda x:int(x.split("_")[1].split(".")[0])
        )[:MAX_FRAMES]

        ppl_imgs=[]
        xpl_imgs=[]

        for f in frames:

            ppl_path=os.path.join(ppl_folder,f)
            xpl_path=os.path.join(xpl_folder,f)

            ppl_img=Image.open(ppl_path).convert("RGB")
            xpl_img=Image.open(xpl_path).convert("RGB")

            ppl_imgs.append(transform(ppl_img))
            xpl_imgs.append(transform(xpl_img))

        ppl_tensor=torch.stack(ppl_imgs)
        xpl_tensor=torch.stack(xpl_imgs)

        out_dir=os.path.join(OUTPUT_PATH,mineral)
        os.makedirs(out_dir,exist_ok=True)

        torch.save({
            "ppl":ppl_tensor,
            "xpl":xpl_tensor,
            "label":classes[mineral],
            "name":sample
        }, os.path.join(out_dir, sample+".pt"))

        print("Saved", sample)

print("DONE")