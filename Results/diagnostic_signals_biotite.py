##############################################################################################################

# © 2026, Ariana Ganea
# Aquesta obra està subjecta a una llicència de Creative Commons Reconeixement-NoComercial-SenseObraDerivada
# 4.0 Internacional (CC BY-NC-ND 4.0). Veure : http://creativecommons.org/licenses/by-nc-nd/4.0/

##############################################################################################################

# CODI PER VISUALITZAR ELS SENYALS DIAGNÒSTICS REALS (PIXELS)

##############################################################################################################

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = "../Dataset/segmented_minerals_tensor"
OUTPUT_FOLDER = "opticalsignal"

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


#############################################
# DATASET
#############################################
class MineralDataset(Dataset):
    def __init__(self, root_dir):
        self.samples, self.labels = [], []
        for mineral, label in {"Biotite": 0, "Hornblende": 1}.items():
            mineral_path = os.path.join(root_dir, mineral)
            if not os.path.exists(mineral_path):
                continue
            for file in sorted(os.listdir(mineral_path)):
                if file.endswith(".pt"):
                    self.samples.append(os.path.join(mineral_path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = torch.load(self.samples[idx], weights_only=False)
        return data["ppl"], data["xpl"], torch.tensor(data["label"])


#############################################
# CONSTRUCCIÓ DELS SENYALS
#############################################
def denormalize(frames):
    """Reverteix la normalització ImageNet del tensor → píxels en [0, 1]."""
    return (frames.float() * IMNET_STD + IMNET_MEAN).clamp(0, 1)


def build_mineral_mask(ppl_frames, xpl_frames, std_threshold=0.015):
    """Màscara binària dels píxels del mineral (H, W) basada en la
    VARIABILITAT TEMPORAL. Els píxels del fons del fitxer segmentat són
    constants al llarg dels 72 fotogrames (std ≈ 0); els del mineral
    varien per pleocroisme (PPL) o extinció (XPL). El llindar s'aplica
    sobre la suma std_PPL + std_XPL per capturar tant uns com els altres."""
    gray_ppl = denormalize(ppl_frames).mean(dim=1)   # (T, H, W)
    gray_xpl = denormalize(xpl_frames).mean(dim=1)
    var_signal = gray_ppl.std(dim=0) + gray_xpl.std(dim=0)
    return var_signal > std_threshold


def mean_mineral_brightness(frames, mask):
    """Brillantor mitjana del mineral a cada fotograma (NO normalitza)."""
    raw = denormalize(frames)                    # (T, 3, H, W)
    gray = raw.mean(dim=1)                       # (T, H, W)
    n = mask.sum().item()
    if n < 1:
        return gray.mean(dim=(1, 2)).numpy()
    return (gray * mask.unsqueeze(0)).sum(dim=(1, 2)).numpy() / n


def circular_skip_diff(x, skip=4):
    """Derivada skip-4 amb padding circular (idèntic al model)."""
    x_t = torch.tensor(x, dtype=torch.float32)
    d = x_t[skip:] - x_t[:-skip]
    d = torch.cat([d[-2:-1], d[-1:], d, d[0:1], d[1:2]])
    return d.numpy()


#############################################
# FIGURA
#############################################
def plot_signals(ppl_int, xpl_int, d1, d2, out_path):
    T = len(ppl_int)
    angles = np.arange(T) * 5
    angles_full = np.concatenate([angles, [360]])
    def close_cycle(x): return np.concatenate([x, [x[0]]])

    ppl_c, xpl_c = close_cycle(ppl_int), close_cycle(xpl_int)
    d1_c, d2_c = close_cycle(d1), close_cycle(d2)

    C_PPL, C_XPL, C_DIFF = "#E07B39", "#4A90C2", "#CFCFCF"
    C_D2, C_TITLE = "#9C2C5E", "#1B3F87"

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.4), facecolor="white")

    # ── Panell superior : intensitats brutes ────────────────────────────────
    ax = axes[0]
    ax.fill_between(angles_full, ppl_c, xpl_c, color=C_DIFF, alpha=0.75,
                    label="|PPL − XPL|", zorder=1)
    ax.plot(angles_full, ppl_c, color=C_PPL, lw=2.1,
            label="PPL_intensity (pleocroisme)", zorder=3)
    ax.plot(angles_full, xpl_c, color=C_XPL, lw=2.1,
            label="XPL_intensity (extinció)", zorder=3)
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 50))
    #ax.set_xlabel("Angle de la platina (graus)", fontsize=10.5)
    ax.set_ylabel("Brillantor mitjana del mineral", fontsize=10.5)
    ax.set_title("Pleocroisme (PPL) i Extinció (XPL)",
                 fontsize=12.5, fontweight="bold", color=C_TITLE, pad=10)
    ax.grid(True, ls=":", color="#BBB", linewidth=0.7, alpha=0.85)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True,
              edgecolor="#CCC", facecolor="white", framealpha=0.95)

    # ── Panell inferior : derivades brutes ──────────────────────────────────
    ax = axes[1]
    ax.axhline(0, color="#888", lw=0.6, zorder=1)
    ax.plot(angles_full, d1_c, color=C_XPL, lw=1.7,
            label="ΔXPL (1a derivada)", zorder=3)
    ax.plot(angles_full, d2_c, color=C_D2, lw=1.7,
            label="Δ²XPL (2a derivada)", zorder=3)
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 50))
    ax.set_xlabel("Angle de la platina (graus)", fontsize=10.5)
    ax.set_ylabel("Magnitud", fontsize=10.5)
    #ax.set_title("Derivades temporals",
    #             fontsize=12.5, fontweight="bold", color=C_TITLE, pad=10)
    ax.grid(True, ls=":", color="#BBB", linewidth=0.7, alpha=0.85)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True,
              edgecolor="#CCC", facecolor="white", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Figura desada: {out_path}")
    plt.show()


#############################################
# MAIN
#############################################
def main():
    dataset = MineralDataset(DATASET_PATH)
    bio_indices = [i for i, l in enumerate(dataset.labels) if l == 0]
    if not bio_indices:
        raise RuntimeError("No s'ha trobat cap mostra de Biotita.")
    idx = bio_indices[3]
    print(f"Mostra escollida: Biotita · idx {idx}\n  {dataset.samples[idx]}")

    ppl_frames, xpl_frames, _ = dataset[idx]
    print(f"Tensor PPL: {tuple(ppl_frames.shape)}  ·  XPL: {tuple(xpl_frames.shape)}")

    # Màscara per VARIABILITAT TEMPORAL — aïlla el mineral del fons constant
    mask = build_mineral_mask(ppl_frames, xpl_frames, std_threshold=0.015)
    print(f"Píxels del mineral: {mask.sum().item():,} / {mask.numel():,} "
          f"({mask.float().mean().item()*100:.1f}%)")

    # Senyals BRUTS — sense normalitzar, sense escalar
    ppl_int = mean_mineral_brightness(ppl_frames, mask)
    xpl_int = mean_mineral_brightness(xpl_frames, mask)
    print(f"PPL_intensity: min={ppl_int.min():.4f}  max={ppl_int.max():.4f}  "
          f"rang={ppl_int.max()-ppl_int.min():.4f}")
    print(f"XPL_intensity: min={xpl_int.min():.4f}  max={xpl_int.max():.4f}  "
          f"rang={xpl_int.max()-xpl_int.min():.4f}")

    d1 = circular_skip_diff(xpl_int, skip=4)
    d2 = circular_skip_diff(d1, skip=4)
    print(f"ΔXPL : min={d1.min():.4f}  max={d1.max():.4f}")
    print(f"Δ²XPL: min={d2.min():.4f}  max={d2.max():.4f}")

    out_path = os.path.join(OUTPUT_FOLDER,
                            "Senyals_diagnostics_Biotita_real.png")
    plot_signals(ppl_int, xpl_int, d1, d2, out_path)


if __name__ == "__main__":
    main()
