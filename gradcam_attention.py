##############################################################################################################

# © 2026, Ariana Ganea
# Aquesta obra està subjecta a una llicència de Creative Commons Reconeixement-NoComercial-SenseObraDerivada
# 4.0 Internacional (CC BY-NC-ND 4.0). Veure : http://creativecommons.org/licenses/by-nc-nd/4.0/

##############################################################################################################

# CODI PER VISUALITZAR GRADCAM AMB EL MODEL :
# RESNET18 + ATTENTION FINAL
# Genera (a) figura PNG amb top-3 frames i (b) GIF animat amb els 72 fotogrames,
# ON EL HEATMAP S'IL·LUMINA / S'APAGA SEGONS EL PES D'ATENCIÓ DEL FRAME
# (Spatio-Temporal Grad-CAM aproximat — atenció temporal × Grad-CAM 2D espacial)

##############################################################################################################

import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models
from sklearn.model_selection import StratifiedKFold
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Circle
from PIL import Image

DATASET_PATH = "Dataset/segmented_minerals_tensor"
RESULTS_DIR = "Models/resnet18_final_attention"
OUTPUT_FOLDER = "Results/gradcam"
FOLD = 1


GAMMA_TEMPORAL = 2.0

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


#############################################
# DATASET I ARQUITECTURA
#############################################
class MineralDataset(Dataset):
    def __init__(self, root_dir, max_frames=72, is_train=False):
        self.samples = []
        self.labels = []
        classes = {"Biotite": 0, "Hornblende": 1}
        for mineral, label in classes.items():
            mineral_path = os.path.join(root_dir, mineral)
            if not os.path.exists(mineral_path): continue
            for file in sorted(os.listdir(mineral_path)):
                if file.endswith(".pt"):
                    self.samples.append(os.path.join(mineral_path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = torch.load(self.samples[idx], weights_only=False)
        return data["ppl"], data["xpl"], torch.tensor(data["label"])


class ResNet18_Attention(nn.Module):
    """ResNet18 + mecanisme d'atenció temporal (model definitiu del TFG)."""

    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet18_Weights
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        # Congelem L1-L3, només L4 entrenable (Green AI)
        for param in self.cnn.parameters(): param.requires_grad = False
        for param in self.cnn[7].parameters(): param.requires_grad = True

        feature_dim = 512  # sortida del backbone ResNet18
        fused_dim = feature_dim * 3 + 5 + 2  # = 1543
        self.bottleneck = nn.Linear(fused_dim, 512)

        self.attention = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, ppl, xpl):
        B, T, C, H, W = ppl.shape
        ppl_flat = ppl.reshape(B * T, C, H, W)
        xpl_flat = xpl.reshape(B * T, C, H, W)

        # Embeddings de cada fotograma via backbone compartit
        ppl_feat = self.cnn(ppl_flat).view(B, T, -1)
        xpl_feat = self.cnn(xpl_flat).view(B, T, -1)

        # Intensitat ponderada amb màscara binaria (proxy de pleocroisme/extinció)
        ppl_mag = torch.norm(ppl_feat, dim=2, keepdim=True)
        xpl_mag = torch.norm(xpl_feat, dim=2, keepdim=True)
        ppl_mask = (ppl_mag > ppl_mag.mean(dim=1, keepdim=True) * 0.5).float()
        xpl_mask = (xpl_mag > xpl_mag.mean(dim=1, keepdim=True) * 0.5).float()
        ppl_intensity = (ppl_feat.mean(dim=2, keepdim=True) * ppl_mask) / (ppl_mask + 1e-6)
        xpl_intensity = (xpl_feat.mean(dim=2, keepdim=True) * xpl_mask) / (xpl_mask + 1e-6)

        # Derivades temporals (Δ amb skip 4) i padding CIRCULAR (la rotació és cíclica)
        delta_ppl = ppl_intensity[:, 4:] - ppl_intensity[:, :-4]
        delta_xpl = xpl_intensity[:, 4:] - xpl_intensity[:, :-4]
        delta_ppl = torch.cat([delta_ppl[:, -2:-1], delta_ppl[:, -1:], delta_ppl,
                               delta_ppl[:, 0:1], delta_ppl[:, 1:2]], dim=1)
        delta_xpl = torch.cat([delta_xpl[:, -2:-1], delta_xpl[:, -1:], delta_xpl,
                               delta_xpl[:, 0:1], delta_xpl[:, 1:2]], dim=1)
        delta2_xpl = delta_xpl[:, 4:] - delta_xpl[:, :-4]
        delta2_xpl = torch.cat([delta2_xpl[:, -2:-1], delta2_xpl[:, -1:], delta2_xpl,
                                delta2_xpl[:, 0:1], delta2_xpl[:, 1:2]], dim=1)

        # Contrast bicanal absolut (ingredient innegociable, secció 3.1)
        diff_feat = torch.abs(ppl_feat - xpl_feat)

        # Codificació posicional sinusoïdal (periodicitat 360°)
        angles = torch.arange(T, device=ppl_feat.device) * (2 * np.pi / T)
        sin_enc = torch.sin(angles).view(1, T, 1).expand(B, T, 1)
        cos_enc = torch.cos(angles).view(1, T, 1).expand(B, T, 1)

        # Concatenació final: 512*3 + 5 + 2 = 1543 dims per fotograma
        features = torch.cat([
            ppl_feat, xpl_feat, diff_feat,
            ppl_intensity, xpl_intensity,
            delta_ppl, delta_xpl, delta2_xpl,
            sin_enc, cos_enc
        ], dim=2)

        features = self.bottleneck(features)
        # Guardem els pesos d'atenció per analitzar-los després (Grad-CAM al temps)
        attn_scores = self.attention(features)
        self.attn_weights = torch.softmax(attn_scores, dim=1)

        context = torch.sum(self.attn_weights * features, dim=1)
        x = torch.relu(self.fc1(context))
        return self.fc2(x)


#############################################
# HELPERS DE VISUALITZACIÓ
#############################################
def tensor_to_img(tensor):
    """Tensor normalitzat ImageNet → imatge RGB [0, 1] visualitzable."""
    img = tensor.cpu().detach().numpy().transpose(1, 2, 0)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    return img


def circular_mask(img, bg=1.0):
    """Aplica una màscara circular centrada per simular el camp de visió
    del microscopi. Fora del cercle → color `bg` (1.0 blanc per defecte).
    Funciona amb float [0,1] o uint8 RGB."""
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    r = min(h, w) / 2.0
    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    out = img.copy()
    bg_val = int(bg * 255) if out.dtype == np.uint8 else bg
    out[~mask] = bg_val
    return out


def black_background_xpl(img, threshold=0.20, bg_value=1.0):
    """Substitueix el fons uniforme del XPL (detectat a partir del color
    de les cantonades) per `bg_value` (per defecte 1.0 = BLANC, simulant
    el camp clar del visor del microscopi). Deixa només els píxels del
    mineral i homogeneïtza la resta perquè la creu d'eixos òptics i el
    perímetre del FoV es vegin contra un fons net."""
    corners = np.stack([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]], axis=0)
    bg_color = corners.mean(axis=0)
    dist = np.linalg.norm(img - bg_color, axis=2)
    bg_mask = dist < threshold
    out = img.copy()
    out[bg_mask] = bg_value
    return out


def style_thin_axes(ax, lw=0.5):
    """Eixos amb línies fines per a un look discret."""
    for s in ax.spines.values():
        s.set_linewidth(lw)
    ax.tick_params(width=lw, length=3)


def draw_optical_axes(ax, color="#000000", lw=0.9, alpha=0.85):
    """Dibuixa la creu d'eixos òptics (polaritzador + analitzador) sobre
    el panell XPL, limitada al cercle de visió. Serveix de referència
    per a llegir l'angle d'extinció respecte al pla de vibració."""
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    cx = (xl[0] + xl[1]) / 2.0
    cy = (yl[0] + yl[1]) / 2.0
    r = min(abs(xl[1] - xl[0]), abs(yl[1] - yl[0])) / 2.0
    ax.plot([cx - r, cx + r], [cy, cy], color=color, lw=lw, alpha=alpha,
            zorder=3)
    ax.plot([cx, cx], [cy - r, cy + r], color=color, lw=lw, alpha=alpha,
            zorder=3)


def draw_fov_circle(ax, color="#000000", lw=0.8, alpha=1.0):
    """Dibuixa el perímetre del camp de visió (FoV) circular del microscopi
    com un cercle fi al voltant del contingut del panell."""
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    cx = (xl[0] + xl[1]) / 2.0
    cy = (yl[0] + yl[1]) / 2.0
    r = min(abs(xl[1] - xl[0]), abs(yl[1] - yl[0])) / 2.0
    ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor=color,
                        linewidth=lw, alpha=alpha, zorder=4))


def compute_overlay(img_ppl, grad, act, attn_factor=1.0):
    """
    Calcula el heatmap Grad-CAM amb modulació temporal: la INTENSITAT del
    CAM (vermell vs blau) s'escala pel pes d'atenció, però el heatmap es
    manté sempre visible. Així, quan attn_factor=0, tota la imatge queda
    coberta pel blau fort de la part inferior del JET (atenció = 0 visualment
    correspon a "frame irrellevant però mostrat com a tal").

    attn_factor ∈ [0, 1]:
      · 1.0  → frame amb atenció màxima: heatmap calent (vermell al pic)
      · 0.0  → frame amb atenció zero: heatmap completament BLAU FORT
    """
    # Grad-CAM 2D clàssic
    w = torch.mean(grad, dim=(1, 2), keepdim=True)
    cam = torch.sum(w * act, dim=0)
    cam = F.relu(cam)
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cam.cpu().detach().numpy()
    cam_resized = cv2.resize(cam, (img_ppl.shape[1], img_ppl.shape[0]))

    # MODULACIÓ TEMPORAL: el CAM s'escala pel pes d'atenció.
    # Quan attn_factor=0, cam_modulated=0 → JET retorna BLAU intens uniforme.
    cam_modulated = cam_resized * attn_factor

    img_uint8 = np.uint8(img_ppl * 255)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_modulated), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blending CONSTANT — el heatmap sempre és visible (també quan atn=0)
    alpha_heatmap = 0.6
    alpha_orig = 1.0 - alpha_heatmap
    overlay = cv2.addWeighted(img_uint8, alpha_orig, heatmap, alpha_heatmap, 0)
    return overlay


def fig_to_pil(fig, dpi=110):
    """Renderitza una figura matplotlib a una imatge PIL (RGB)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def compute_attn_factor(pesos_atencio, idx, gamma=GAMMA_TEMPORAL):
    """
    Calcula el factor temporal normalitzat per al frame idx.
    Aplica un gamma > 1 per exagerar el contrast (només els TOP frames brillen).
    """
    attn_max = pesos_atencio.max()
    if attn_max <= 0:
        return 0.0
    return float((pesos_atencio[idx] / attn_max) ** gamma)


#############################################
# RENDER DE LA FIGURA ESTÀTICA (TOP-3 FRAMES)
#############################################
def render_top3_figure(ppl_frames, xpl_frames, grad_all, act_all,
                       pesos_atencio, true_name, out_path):
    """PNG amb els tres fotogrames amb més pes d'atenció."""
    top_3 = np.argsort(pesos_atencio)[-3:][::-1]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), facecolor="white")
    fig.suptitle(f"Anàlisi d'Interpretabilitat (ResNet18+Attention): {true_name}",
                 fontsize=17, fontweight="bold", y=0.98)

    for row, f_idx in enumerate(top_3):
        pes = pesos_atencio[f_idx] * 100
        angle = f_idx * 5
        img_ppl = tensor_to_img(ppl_frames[f_idx])
        img_xpl = tensor_to_img(xpl_frames[f_idx])

        # Modulació temporal del Grad-CAM
        attn_factor = compute_attn_factor(pesos_atencio, f_idx)
        overlay = compute_overlay(img_ppl, grad_all[f_idx], act_all[f_idx],
                                  attn_factor=attn_factor)

        axes[row, 0].imshow(circular_mask(img_ppl))
        axes[row, 0].set_title(f"Angle: {angle}º (PPL)\nPes atenció: {pes:.1f}%",
                               fontsize=12, fontweight="bold")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(circular_mask(img_xpl))
        axes[row, 1].set_title("XPL", fontsize=12, fontweight="bold")
        axes[row, 1].axis("off")
        draw_optical_axes(axes[row, 1])
        draw_fov_circle(axes[row, 1])

        axes[row, 2].imshow(circular_mask(overlay))
        axes[row, 2].set_title("Grad-CAM",
                               fontsize=12, fontweight="bold")
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figura PNG: {out_path}")


#############################################
# RENDER DEL GIF ANIMAT (72 FRAMES)
#############################################
def render_gif(ppl_frames, xpl_frames, grad_all, act_all,
               pesos_atencio, true_name, out_path, fps=8):
    """GIF amb PPL, XPL, Grad-CAM modulat i corba d'atenció normalitzada (0–1)."""
    T = ppl_frames.shape[0]

    overlays = []
    for t in range(T):
        img_ppl = tensor_to_img(ppl_frames[t])
        attn_factor = compute_attn_factor(pesos_atencio, t)
        ov = compute_overlay(img_ppl, grad_all[t], act_all[t],
                             attn_factor=attn_factor)
        overlays.append(circular_mask(ov))

    angles_deg = np.arange(T) * 5
    attn_max = pesos_atencio.max()
    pesos_norm = (pesos_atencio / attn_max) ** GAMMA_TEMPORAL if attn_max > 0 \
        else np.zeros_like(pesos_atencio)

    pil_frames = []
    for t in range(T):
        fig = plt.figure(figsize=(13, 7.5), facecolor="white")
        gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1],
                               hspace=0.40, wspace=0.10)

        ax_ppl = fig.add_subplot(gs[0, 0])
        ax_xpl = fig.add_subplot(gs[0, 1])
        ax_cam = fig.add_subplot(gs[0, 2])
        ax_mod = fig.add_subplot(gs[1, :])

        angle = t * 5
        mod_pct = pesos_norm[t] * 100
        fig.suptitle(f"ResNet18 + Attention — {true_name} · "
                     f"frame {t + 1}/72 ({angle}°) · modulació {mod_pct:.0f}%",
                     fontsize=13, fontweight="bold", y=0.97)

        # Panells superiors
        ax_ppl.imshow(circular_mask(tensor_to_img(ppl_frames[t])));
        ax_ppl.axis("off")
        ax_ppl.set_title("PPL", fontsize=11, fontweight="bold")
        ax_xpl.imshow(circular_mask(tensor_to_img(xpl_frames[t])));
        ax_xpl.axis("off")
        ax_xpl.set_title("XPL", fontsize=11, fontweight="bold")
        draw_optical_axes(ax_xpl)
        draw_fov_circle(ax_xpl)
        ax_cam.imshow(overlays[t]);
        ax_cam.axis("off")
        ax_cam.set_title("Grad-CAM",
                         fontsize=11, fontweight="bold")

        # Panell temporal: només atenció normalitzada
        ax_mod.plot(angles_deg, pesos_norm, color="#9E2A2B", lw=1.4)
        ax_mod.fill_between(angles_deg, pesos_norm, alpha=0.22, color="#9E2A2B")
        ax_mod.axvline(angle, color="#C0392B", lw=1.4)
        ax_mod.set_xlim(0, 355)
        ax_mod.set_ylim(0, 1.05)
        ax_mod.set_xlabel("Rotació (graus)", fontsize=10.5, fontweight="bold")
        ax_mod.tick_params(axis="x", top=False, labeltop=False,
                           bottom=True, labelbottom=True)
        ax_mod.set_title("Atenció normalitzada",
                         fontsize=11.5, fontweight="bold", pad=12)
        ax_mod.grid(True, ls="--", alpha=0.4, linewidth=0.4)
        for s in ["top", "right"]:
            ax_mod.spines[s].set_visible(False)
        style_thin_axes(ax_mod, lw=0.5)

        pil_frames.append(fig_to_pil(fig, dpi=100))

    duration_ms = int(1000 / fps)
    pil_frames[0].save(out_path, save_all=True, append_images=pil_frames[1:],
                       duration=duration_ms, loop=0, optimize=True)
    print(f"  GIF animat: {out_path}")

def plot_min_max_comparison(ppl_frames, xpl_frames, grad_all, act_all,
                               pesos_atencio, true_name, min_angle_override=None,
                               out_path=None):
    """
    Figura amb dues columnes (XPL + Grad-CAM) per als moments de MÀXIMA i
    MÍNIMA atenció. L'XPL es mostra amb el fons del cercle en NEGRE
    (camp obscur del microscopi entre polaritzadors creuats) i la creu
    d'eixos òptics en blanc com a referència de l'angle d'extinció.

    Si es passa `min_angle_override` (en graus), el frame de MÍNIM
    s'ancorarà a aquest angle en lloc del mínim absolut de la corba.
    """
    attn_max = pesos_atencio.max()
    pesos_norm = (pesos_atencio / attn_max) ** GAMMA_TEMPORAL if attn_max > 0 \
                  else np.zeros_like(pesos_atencio)

    idx_max = int(np.argmax(pesos_atencio))
    if min_angle_override is not None:
        idx_min = int(min_angle_override // 5)
    else:
        idx_min = int(np.argmin(pesos_atencio))

    angles_deg = np.arange(len(pesos_atencio)) * 5
    angle_max = idx_max * 5
    angle_min = idx_min * 5
    y_max = pesos_norm[idx_max]
    y_min = pesos_norm[idx_min]

    COLOR_MAX = "#1F3A6E"
    COLOR_MIN = "#7F7F7F"

    # Posicionament dinàmic: el grup amb menor angle queda a l'esquerra
    if angle_max <= angle_min:
        left_kind, right_kind = "MAX", "MIN"
        left_idx, right_idx = idx_max, idx_min
    else:
        left_kind, right_kind = "MIN", "MAX"
        left_idx, right_idx = idx_min, idx_max

    fig = plt.figure(figsize=(14, 7.5), facecolor="white")
    # 2 cols (XPL + GradCAM) · separador · 2 cols (XPL + GradCAM)
    gs = gridspec.GridSpec(2, 5, height_ratios=[4.5, 1.4],
                            width_ratios=[1, 1, 0.35, 1, 1],
                            hspace=0.05, wspace=0.10,
                            top=0.91, bottom=0.08)

    fig.suptitle(true_name, fontsize=15, fontweight="bold", y=0.97)

    def render_group(start_col, kind, f_idx):
        color = COLOR_MAX if kind == "MAX" else COLOR_MIN
        label = "MÀXIM" if kind == "MAX" else "MÍNIM"
        axes_g = [fig.add_subplot(gs[0, start_col + i]) for i in range(2)]

        img_xpl = tensor_to_img(xpl_frames[f_idx])
        img_ppl_for_cam = tensor_to_img(ppl_frames[f_idx])
        attn_factor = compute_attn_factor(pesos_atencio, f_idx)
        overlay = compute_overlay(img_ppl_for_cam, grad_all[f_idx], act_all[f_idx],
                                  attn_factor=attn_factor)

        # XPL: fons del FoV en BLANC + creu i perímetre negres
        xpl_white = black_background_xpl(img_xpl)
        axes_g[0].imshow(circular_mask(xpl_white, bg=1.0)); axes_g[0].axis("off")
        axes_g[0].set_title(f"XPL — {label}\nangle {f_idx*5}°  ·  "
                            f"modulació {attn_factor*100:.0f}%",
                            fontsize=12, fontweight="bold", color=color)
        draw_optical_axes(axes_g[0])
        draw_fov_circle(axes_g[0])

        # Grad-CAM: igual que sempre (fons blanc fora del cercle)
        axes_g[1].imshow(circular_mask(overlay)); axes_g[1].axis("off")
        axes_g[1].set_title("Grad-CAM", fontsize=12, fontweight="bold")
        return axes_g

    render_group(0, left_kind, left_idx)
    render_group(3, right_kind, right_idx)

    # Corba inferior — tema clar
    ax_curve = fig.add_subplot(gs[1, :])
    ax_curve.plot(angles_deg, pesos_norm, color="#9E2A2B", lw=1.4)
    ax_curve.fill_between(angles_deg, pesos_norm, alpha=0.22, color="#9E2A2B")

    ax_curve.plot(angle_max, y_max, "o", color=COLOR_MAX, markersize=12,
                  markeredgecolor="white", markeredgewidth=1.6, zorder=6,
                  label=f"MÀXIM ({angle_max}°)")
    ax_curve.plot(angle_min, y_min, "o", color=COLOR_MIN, markersize=12,
                  markeredgecolor="white", markeredgewidth=1.6, zorder=6,
                  label=f"MÍNIM ({angle_min}°)")

    ax_curve.set_xlim(0, 355)
    ax_curve.set_ylim(0, 1.15)
    ax_curve.set_xlabel("Rotació (graus)", fontsize=11, fontweight="bold")
    ax_curve.tick_params(axis="x", top=False, labeltop=False,
                         bottom=True, labelbottom=True)
    ax_curve.set_title("Atenció normalitzada",
                       fontsize=12, fontweight="bold", pad=10)
    ax_curve.grid(True, ls="--", alpha=0.4, linewidth=0.4)
    for s in ["top", "right"]:
        ax_curve.spines[s].set_visible(False)
    style_thin_axes(ax_curve, lw=0.5)
    ax_curve.legend(loc="upper center", fontsize=10, frameon=True,
                    edgecolor="#CCC", facecolor="white", ncol=2)

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Figura PNG: {out_path}")
    else:
        plt.show()


def plot_normalized_attention_only(pesos_atencio, true_name):
    """
    Figura rectangular compacta (5×3 polzades) que mostra NOMÉS la corba
    d'atenció normalitzada per a una mostra. Eix X (graus) i ticks a baix.
    """
    attn_max = pesos_atencio.max()
    pesos_norm = (pesos_atencio / attn_max) ** GAMMA_TEMPORAL if attn_max > 0 \
        else np.zeros_like(pesos_atencio)
    angles_deg = np.arange(len(pesos_atencio)) * 5

    fig, ax = plt.subplots(figsize=(5, 3), facecolor="white")
    ax.plot(angles_deg, pesos_norm, color="#9E2A2B", lw=1.4)
    ax.fill_between(angles_deg, pesos_norm, alpha=0.22, color="#9E2A2B")
    ax.set_xlim(0, 355)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Rotació (graus)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Atenció normalitzada", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", top=False, labeltop=False,
                   bottom=True, labelbottom=True)
    ax.set_title(f"Atenció normalitzada — {true_name}",
                 fontsize=11, fontweight="bold", pad=8)
    ax.grid(True, ls="--", alpha=0.4, linewidth=0.4)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    style_thin_axes(ax, lw=0.5)

    plt.tight_layout()
    plt.show()


#############################################
# MAIN
#############################################
def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Utilitzant dispositiu: {DEVICE}")
    print(f"Dataset:    {DATASET_PATH}")
    print(f"Checkpoint: {RESULTS_DIR}")
    print(f"Sortida:    {OUTPUT_FOLDER}")
    print(f"Gamma temporal: {GAMMA_TEMPORAL}\n")

    if not os.path.isdir(DATASET_PATH):
        raise FileNotFoundError(f"No existeix el dataset: {DATASET_PATH}")

    dataset = MineralDataset(DATASET_PATH, max_frames=72, is_train=False)
    print(f"Mostres: {len(dataset)}  "
          f"(biotita: {dataset.labels.count(0)}, hornblenda: {dataset.labels.count(1)})\n")

    # ── Errors coneguts del model LSTM ───────────────────────────────────────
    LSTM_FP = {
        "I08_rot2_crystal_1", "green_schist_rot2_crystal_3", "I08_rot2_crystal_3",
        "quartz_monzonite_jersey_rot2_crystal_2", "quartz_monzonite_jersey_rot2_crystal_4",
    }
    LSTM_FN = {
        "M01_rot2_crystal_1", "I16_rot2_crystal_2",
        "amygdaloidal_monchiquite_rot1_crystal_2",
        "epidote_amphibolite_mayo_rot2_crystal_2",
        "granophyre_isle_rum_rot1_crystal_4",
    }

    name_to_idx = {
        os.path.splitext(os.path.basename(p))[0]: i
        for i, p in enumerate(dataset.samples)
    }

    labels = dataset.labels
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(range(len(dataset)), labels))
    idx_to_fold = {}
    for fn, (_, tidx) in enumerate(splits, start=1):
        for i in tidx:
            idx_to_fold[i] = fn

    # Model cache
    loaded_fold = {}

    def get_model(fold_num):
        if fold_num not in loaded_fold:
            ckpt = os.path.join(RESULTS_DIR, f"fold_{fold_num}", "best_model.pth")
            if not os.path.isfile(ckpt):
                raise FileNotFoundError(f"Checkpoint no trobat: {ckpt}")
            m = ResNet18_Attention().to(DEVICE)
            m.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model_state_dict"])
            m.eval()
            loaded_fold[fold_num] = m
        return loaded_fold[fold_num]

    # ── Passada 1: errors del model Attention (tots els 5 folds) ─────────────
    print("── Passada 1: errors del model Attention ──")
    ATTN_FP, ATTN_FN = set(), set()
    with torch.no_grad():
        for fold_num in range(1, 6):
            try:
                m = get_model(fold_num)
            except FileNotFoundError:
                print(f"  Fold {fold_num}: checkpoint no trobat, saltat.")
                continue
            _, test_idx = splits[fold_num - 1]
            for idx in test_idx:
                ppl_f, xpl_f, true_lbl = dataset[idx]
                out   = m(ppl_f.unsqueeze(0).to(DEVICE),
                          xpl_f.unsqueeze(0).to(DEVICE))
                pred  = torch.argmax(out, dim=1).item()
                label = true_lbl.item()
                name  = os.path.splitext(os.path.basename(dataset.samples[idx]))[0]
                if label == 0 and pred == 1:
                    ATTN_FP.add(name)
                elif label == 1 and pred == 0:
                    ATTN_FN.add(name)

    print(f"  Attention FP: {len(ATTN_FP)}  FN: {len(ATTN_FN)}\n")

    # ── Categorització ────────────────────────────────────────────────────────
    both_fp      = LSTM_FP & ATTN_FP
    both_fn      = LSTM_FN & ATTN_FN
    attn_only_fp = ATTN_FP - LSTM_FP
    attn_only_fn = ATTN_FN - LSTM_FN
    lstm_only_fp = LSTM_FP - ATTN_FP
    lstm_only_fn = LSTM_FN - ATTN_FN

    def prefix_for(name):
        if name in both_fp:      return "BOTH_FP"
        if name in both_fn:      return "BOTH_FN"
        if name in attn_only_fp: return "ATTN_FP"
        if name in attn_only_fn: return "ATTN_FN"
        if name in lstm_only_fp: return "LSTM_FP"
        if name in lstm_only_fn: return "LSTM_FN"
        return "UNKNOWN"

    all_errors = LSTM_FP | LSTM_FN | ATTN_FP | ATTN_FN

    # ── GradCAM helper ────────────────────────────────────────────────────────
    def run_gradcam(idx, fold_num):
        m = get_model(fold_num)
        ppl_frames, xpl_frames, _ = dataset[idx]
        ppl = ppl_frames.unsqueeze(0).to(DEVICE).requires_grad_(True)
        xpl = xpl_frames.unsqueeze(0).to(DEVICE).requires_grad_(True)
        act_buf, grad_buf = [None], [None]

        def fw_hook(mod, i, o):  act_buf[0] = o
        def bw_hook(mod, gi, go): grad_buf[0] = go[0]

        h1 = m.cnn[7].register_forward_hook(fw_hook)
        h2 = m.cnn[7].register_full_backward_hook(bw_hook)
        out = m(ppl, xpl)
        pred_class = torch.argmax(out, dim=1).item()
        m.zero_grad()
        out[0, pred_class].backward()
        h1.remove(); h2.remove()
        pesos = m.attn_weights[0].cpu().detach().numpy().flatten()
        return ppl_frames, xpl_frames, grad_buf[0], act_buf[0], pesos

    # ── Passada 2: PNG + GIF per a tots els errors ────────────────────────────
    print(f"── Passada 2: {len(all_errors)} mostres — PNG + GIF ──")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for name in sorted(all_errors):
        if name not in name_to_idx:
            print(f"  [AVÍS] No trobat: {name}")
            continue
        idx      = name_to_idx[name]
        fold_num = idx_to_fold.get(idx, 1)
        prefix   = prefix_for(name)
        label    = dataset.labels[idx]
        cls_name = "Biotita" if label == 0 else "Hornblenda"
        print(f"\n→ {prefix} ({cls_name}) fold {fold_num}: {name}")

        ppl_f, xpl_f, grad_ppl, act_ppl, pesos = run_gradcam(idx, fold_num)
        print(f"  Atenció — max: {pesos.max()*100:.2f}%  "
              f"min: {pesos.min()*100:.2f}%  "
              f"frame màx: {int(np.argmax(pesos))*5}°")

        png_path = os.path.join(OUTPUT_FOLDER, f"{prefix}_{name}.png")
        gif_path = os.path.join(OUTPUT_FOLDER, f"{prefix}_{name}.gif")
        plot_min_max_comparison(ppl_f, xpl_f, grad_ppl, act_ppl,
                                pesos, name, out_path=png_path)
        render_gif(ppl_f, xpl_f, grad_ppl, act_ppl,
                   pesos, name, gif_path, fps=8)

    # ── Resum final ───────────────────────────────────────────────────────────
    abs_output = os.path.abspath(OUTPUT_FOLDER)

    def print_entry(prefix, name):
        png = os.path.join(abs_output, f"{prefix}_{name}.png")
        gif = os.path.join(abs_output, f"{prefix}_{name}.gif")
        print(f"  • {name}")
        print(f"      PNG: file://{png}")
        print(f"      GIF: file://{gif}")

    print("\n" + "═" * 60)
    print("RESUM D'ERRORS")
    print("═" * 60)

    print(f"\n[AMBDÓS MODELS] FP — biotita → hornblenda ({len(both_fp)}):")
    for n in sorted(both_fp):   print_entry("BOTH_FP", n)
    print(f"\n[AMBDÓS MODELS] FN — hornblenda → biotita ({len(both_fn)}):")
    for n in sorted(both_fn):   print_entry("BOTH_FN", n)

    print(f"\n[SOLO ATTENTION] FP ({len(attn_only_fp)}):")
    for n in sorted(attn_only_fp): print_entry("ATTN_FP", n)
    print(f"\n[SOLO ATTENTION] FN ({len(attn_only_fn)}):")
    for n in sorted(attn_only_fn): print_entry("ATTN_FN", n)

    print(f"\n[SOLO LSTM] FP ({len(lstm_only_fp)}):")
    for n in sorted(lstm_only_fp): print_entry("LSTM_FP", n)
    print(f"\n[SOLO LSTM] FN ({len(lstm_only_fn)}):")
    for n in sorted(lstm_only_fn): print_entry("LSTM_FN", n)

    print(f"\n── Fet. {len(all_errors)} figures generades a {OUTPUT_FOLDER} ──")


if __name__ == "__main__":
    main()
