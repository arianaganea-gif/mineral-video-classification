##############################################################################################################

# © 2026, Ariana Ganea
# Aquesta obra està subjecta a una llicència de Creative Commons Reconeixement-NoComercial-SenseObraDerivada
# 4.0 Internacional (CC BY-NC-ND 4.0). Veure : http://creativecommons.org/licenses/by-nc-nd/4.0/

##############################################################################################################

# SCRIPT UNIFICAT D'ENTRENAMENT — 7 MODELS DEL TFG
#
# Per pantalla l'usuari tria quin model entrenar:
#   [1] ResNet18 + Attention      [4] ResNet50 + LSTM         [7] ResNet18 + StaticImage
#   [2] ResNet50 + Attention      [5] ResNet18 + MeanPooling  [0] Sortir
#   [3] ResNet18 + LSTM           [6] ResNet50 + MeanPooling
#

##############################################################################################################

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.model_selection import StratifiedKFold


#############################################
# CONFIGURACIÓ GLOBAL
#############################################

DATASET_PATH = "Dataset/segmented_minerals_tensor"
BATCH_SIZE = 4
MAX_FRAMES = 72
LR = 1e-4

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = "cpu"


def seed_everything(seed=42):
    """Fixa totes les llavors per garantir reproductibilitat."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


#############################################
# AUGMENTACIONS
#############################################

def augment_video_cpu(ppl, xpl):
    """Augmentació al __getitem__ (versió ResNet50). Tensor 4D (T,C,H,W)."""
    shift = random.randint(0, len(ppl) - 1)
    ppl = torch.roll(ppl, shift, 0)
    xpl = torch.roll(xpl, shift, 0)

    if random.random() < 0.5:
        factor = random.uniform(0.9, 1.1)
        ppl = torch.clamp(ppl * factor, 0, 1)
        xpl = torch.clamp(xpl * factor, 0, 1)

    if random.random() < 0.5:
        ppl = torch.flip(ppl, dims=[3])
        xpl = torch.flip(xpl, dims=[3])

    if random.random() < 0.5:
        ppl = torch.flip(ppl, dims=[2])
        xpl = torch.flip(xpl, dims=[2])

    if random.random() < 0.3:
        noise = torch.randn_like(ppl) * 0.01
        ppl = torch.clamp(ppl + noise, 0, 1)
        xpl = torch.clamp(xpl + noise, 0, 1)

    return ppl, xpl


def augment_video_gpu(ppl, xpl):
    """Augmentació al loop d'entrenament (versió ResNet18). Tensor 5D (B,T,C,H,W)."""
    B, T, C, H, W = ppl.shape

    shift = random.randint(0, T - 1)
    ppl = torch.roll(ppl, shift, dims=1)
    xpl = torch.roll(xpl, shift, dims=1)

    if random.random() < 0.5:
        factor = random.uniform(0.9, 1.1)
        ppl = torch.clamp(ppl * factor, 0, 1)
        xpl = torch.clamp(xpl * factor, 0, 1)

    if random.random() < 0.5:
        ppl = torch.flip(ppl, dims=[3])
        xpl = torch.flip(xpl, dims=[3])

    if random.random() < 0.5:
        ppl = torch.flip(ppl, dims=[4])
        xpl = torch.flip(xpl, dims=[4])

    if random.random() < 0.3:
        noise = torch.randn_like(ppl) * 0.01
        ppl = torch.clamp(ppl + noise, 0, 1)
        xpl = torch.clamp(xpl + noise, 0, 1)

    return ppl, xpl


def augment_static_gpu(ppl, xpl):
    """Augmentació espacial per al model d'imatge estàtica. Tensor 4D (B,C,H,W)."""
    if random.random() < 0.5:
        factor = random.uniform(0.9, 1.1)
        ppl = torch.clamp(ppl * factor, 0, 1)
        xpl = torch.clamp(xpl * factor, 0, 1)

    if random.random() < 0.5:
        ppl = torch.flip(ppl, dims=[2])
        xpl = torch.flip(xpl, dims=[2])

    if random.random() < 0.5:
        ppl = torch.flip(ppl, dims=[3])
        xpl = torch.flip(xpl, dims=[3])

    if random.random() < 0.3:
        noise = torch.randn_like(ppl) * 0.01
        ppl = torch.clamp(ppl + noise, 0, 1)
        xpl = torch.clamp(xpl + noise, 0, 1)

    return ppl, xpl


#############################################
# DATASETS
#############################################

class MineralDataset(Dataset):
    """Dataset de vídeos sencers (PPL+XPL).

    cache=True precarga tots els tensors a RAM (ResNet18); cache=False carrega sota demanda
    (ResNet50).
    """

    def __init__(self, root_dir, is_train, cache=False,
                 augment_in_getitem=False, sort_files=True):
        self.samples = []
        self.labels = []
        self.is_train = is_train
        self.augment_in_getitem = augment_in_getitem
        self._cache = {} if cache else None

        classes = {"Biotite": 0, "Hornblende": 1}
        for mineral, lbl in classes.items():
            mineral_path = os.path.join(root_dir, mineral)
            if not os.path.exists(mineral_path):
                continue
            files = os.listdir(mineral_path)
            if sort_files:
                files = sorted(files)
            for f in files:
                if f.endswith(".pt"):
                    self.samples.append(os.path.join(mineral_path, f))
                    self.labels.append(lbl)

        if cache:
            print(f"Precarregant {len(self.samples)} mostres a la memòria RAM...")
            for i, p in enumerate(self.samples):
                self._cache[i] = torch.load(p, weights_only=False)
            print("Dataset carregat correctament.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self._cache[idx] if self._cache is not None else torch.load(self.samples[idx])
        ppl = data["ppl"]
        xpl = data["xpl"]
        label = torch.tensor(data["label"])

        if self.augment_in_getitem and self.is_train:
            ppl, xpl = augment_video_cpu(ppl, xpl)

        return ppl, xpl, label


class MineralDatasetSingleFrame(Dataset):
    """Dataset de frame únic per al model d'imatge estàtica.

    Train: agafa un frame aleatori dels 72. Test: sempre el frame 0.
    """

    def __init__(self, root_dir, is_train):
        self.samples = []
        self.labels = []
        self.is_train = is_train
        self._cache = {}

        classes = {"Biotite": 0, "Hornblende": 1}
        for mineral, lbl in classes.items():
            mineral_path = os.path.join(root_dir, mineral)
            if not os.path.exists(mineral_path):
                continue
            for f in sorted(os.listdir(mineral_path)):
                if f.endswith(".pt"):
                    self.samples.append(os.path.join(mineral_path, f))
                    self.labels.append(lbl)

        print(f"Precarregant {len(self.samples)} mostres a la memòria RAM...")
        for i, p in enumerate(self.samples):
            self._cache[i] = torch.load(p, weights_only=False)
        print("Dataset carregat correctament.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self._cache[idx]
        ppl = data["ppl"]
        xpl = data["xpl"]
        label = torch.tensor(data["label"])

        frame_idx = random.randint(0, ppl.shape[0] - 1) if self.is_train else 0
        return ppl[frame_idx], xpl[frame_idx], label


##############################################################################
# ARQUITECTURES
##############################################################################

# ----- ResNet18 + Attention -----
class ResNet18_Attention(nn.Module):
    """ResNet18 com a backbone + atenció temporal sobre la sequència de 72 frames."""

    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet18_Weights
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.cnn.parameters():
            p.requires_grad = False
        for p in self.cnn[7].parameters():
            p.requires_grad = True

        fused_dim = 512 * 3 + 5 + 2
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
        ppl_feat = self.cnn(ppl.reshape(B * T, C, H, W)).view(B, T, -1)
        xpl_feat = self.cnn(xpl.reshape(B * T, C, H, W)).view(B, T, -1)

        ppl_mag = torch.norm(ppl_feat, dim=2, keepdim=True)
        xpl_mag = torch.norm(xpl_feat, dim=2, keepdim=True)
        ppl_mask = (ppl_mag > ppl_mag.mean(dim=1, keepdim=True) * 0.5).float()
        xpl_mask = (xpl_mag > xpl_mag.mean(dim=1, keepdim=True) * 0.5).float()
        ppl_intensity = (ppl_feat.mean(dim=2, keepdim=True) * ppl_mask) / (ppl_mask + 1e-6)
        xpl_intensity = (xpl_feat.mean(dim=2, keepdim=True) * xpl_mask) / (xpl_mask + 1e-6)

        delta_ppl = ppl_intensity[:, 4:] - ppl_intensity[:, :-4]
        delta_xpl = xpl_intensity[:, 4:] - xpl_intensity[:, :-4]
        delta_ppl = torch.cat([delta_ppl[:, -2:-1], delta_ppl[:, -1:], delta_ppl,
                               delta_ppl[:, 0:1], delta_ppl[:, 1:2]], dim=1)
        delta_xpl = torch.cat([delta_xpl[:, -2:-1], delta_xpl[:, -1:], delta_xpl,
                               delta_xpl[:, 0:1], delta_xpl[:, 1:2]], dim=1)
        delta2_xpl = delta_xpl[:, 4:] - delta_xpl[:, :-4]
        delta2_xpl = torch.cat([delta2_xpl[:, -2:-1], delta2_xpl[:, -1:], delta2_xpl,
                                delta2_xpl[:, 0:1], delta2_xpl[:, 1:2]], dim=1)

        diff_feat = torch.abs(ppl_feat - xpl_feat)
        angles = torch.arange(T, device=ppl_feat.device) * (2 * np.pi / T)
        sin_enc = torch.sin(angles).view(1, T, 1).expand(B, T, 1)
        cos_enc = torch.cos(angles).view(1, T, 1).expand(B, T, 1)

        features = torch.cat([
            ppl_feat, xpl_feat, diff_feat, ppl_intensity, xpl_intensity,
            delta_ppl, delta_xpl, delta2_xpl, sin_enc, cos_enc
        ], dim=2)
        features = self.bottleneck(features)
        weights = torch.softmax(self.attention(features), dim=1)
        context = torch.sum(weights * features, dim=1)
        x = torch.relu(self.fc1(context))
        return self.fc2(x)


# ----- ResNet50 + Attention -----
class ResNet50_Attention(nn.Module):
    """ResNet50 com a backbone + atenció temporal."""

    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet50_Weights
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.cnn.parameters():
            p.requires_grad = False
        for p in self.cnn[7].parameters():
            p.requires_grad = True

        fused_dim = 2048 * 3 + 7
        self.bottleneck = nn.Linear(fused_dim, 512)
        self.attention = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.fc1 = nn.Linear(512, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, ppl, xpl):
        B, T, C, H, W = ppl.shape
        ppl_feat = self.cnn(ppl.reshape(B * T, C, H, W)).view(B, T, -1)
        xpl_feat = self.cnn(xpl.reshape(B * T, C, H, W)).view(B, T, -1)

        ppl_mag = torch.norm(ppl_feat, dim=2, keepdim=True)
        xpl_mag = torch.norm(xpl_feat, dim=2, keepdim=True)
        ppl_mask = (ppl_mag > ppl_mag.mean(dim=1, keepdim=True) * 0.5).float()
        xpl_mask = (xpl_mag > xpl_mag.mean(dim=1, keepdim=True) * 0.5).float()
        ppl_intensity = (ppl_feat.mean(dim=2, keepdim=True) * ppl_mask) / (ppl_mask + 1e-6)
        xpl_intensity = (xpl_feat.mean(dim=2, keepdim=True) * xpl_mask) / (xpl_mask + 1e-6)

        d_ppl = ppl_intensity[:, 4:] - ppl_intensity[:, :-4]
        d_xpl = xpl_intensity[:, 4:] - xpl_intensity[:, :-4]
        d_ppl = torch.cat([d_ppl[:, 0:1], d_ppl[:, 0:1], d_ppl,
                           d_ppl[:, -1:], d_ppl[:, -1:]], dim=1)
        d_xpl = torch.cat([d_xpl[:, 0:1], d_xpl[:, 0:1], d_xpl,
                           d_xpl[:, -1:], d_xpl[:, -1:]], dim=1)
        d2_xpl = d_xpl[:, 4:] - d_xpl[:, :-4]
        d2_xpl = torch.cat([d2_xpl[:, 0:1], d2_xpl[:, 0:1], d2_xpl,
                            d2_xpl[:, -1:], d2_xpl[:, -1:]], dim=1)

        diff_feat = torch.abs(ppl_feat - xpl_feat)
        features = torch.cat([
            ppl_feat, xpl_feat, diff_feat, ppl_intensity, xpl_intensity,
            d_ppl, d_xpl, d2_xpl
        ], dim=2)

        angles = torch.arange(T, device=features.device) * (2 * np.pi / T)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(2).repeat(B, 1, 1)
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(2).repeat(B, 1, 1)
        features = torch.cat([features, sin, cos], dim=2)

        features = torch.relu(self.bottleneck(features))
        weights = torch.softmax(self.attention(features), dim=1)
        context = torch.sum(weights * features, dim=1)
        x = torch.relu(self.fc1(context))
        x = self.dropout(x)
        return self.fc2(x)


# ----- ResNet18 + LSTM -----
class ResNet18_LSTM(nn.Module):
    """ResNet18 com a backbone + LSTM (1 capa, hidden 512) per modelar la sequència."""

    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet18_Weights
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.cnn.parameters():
            p.requires_grad = False
        for p in self.cnn[7].parameters():
            p.requires_grad = True

        fused_dim = 512 * 3 + 5 + 2
        self.bottleneck = nn.Linear(fused_dim, 512)
        self.lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, ppl, xpl):
        B, T, C, H, W = ppl.shape
        ppl_feat = self.cnn(ppl.reshape(B * T, C, H, W)).view(B, T, -1)
        xpl_feat = self.cnn(xpl.reshape(B * T, C, H, W)).view(B, T, -1)

        ppl_mag = torch.norm(ppl_feat, dim=2, keepdim=True)
        xpl_mag = torch.norm(xpl_feat, dim=2, keepdim=True)
        ppl_mask = (ppl_mag > ppl_mag.mean(dim=1, keepdim=True) * 0.5).float()
        xpl_mask = (xpl_mag > xpl_mag.mean(dim=1, keepdim=True) * 0.5).float()
        ppl_intensity = (ppl_feat.mean(dim=2, keepdim=True) * ppl_mask) / (ppl_mask + 1e-6)
        xpl_intensity = (xpl_feat.mean(dim=2, keepdim=True) * xpl_mask) / (xpl_mask + 1e-6)

        delta_ppl = ppl_intensity[:, 4:] - ppl_intensity[:, :-4]
        delta_xpl = xpl_intensity[:, 4:] - xpl_intensity[:, :-4]
        delta_ppl = torch.cat([delta_ppl[:, -2:-1], delta_ppl[:, -1:], delta_ppl,
                               delta_ppl[:, 0:1], delta_ppl[:, 1:2]], dim=1)
        delta_xpl = torch.cat([delta_xpl[:, -2:-1], delta_xpl[:, -1:], delta_xpl,
                               delta_xpl[:, 0:1], delta_xpl[:, 1:2]], dim=1)
        delta2_xpl = delta_xpl[:, 4:] - delta_xpl[:, :-4]
        delta2_xpl = torch.cat([delta2_xpl[:, -2:-1], delta2_xpl[:, -1:], delta2_xpl,
                                delta2_xpl[:, 0:1], delta2_xpl[:, 1:2]], dim=1)

        diff_feat = torch.abs(ppl_feat - xpl_feat)
        angles = torch.arange(T, device=ppl_feat.device) * (2 * np.pi / T)
        sin_enc = torch.sin(angles).view(1, T, 1).expand(B, T, 1)
        cos_enc = torch.cos(angles).view(1, T, 1).expand(B, T, 1)

        features = torch.cat([
            ppl_feat, xpl_feat, diff_feat, ppl_intensity, xpl_intensity,
            delta_ppl, delta_xpl, delta2_xpl, sin_enc, cos_enc
        ], dim=2)
        features = self.bottleneck(features)
        _, (hn, _) = self.lstm(features)
        context = hn[-1]
        x = torch.relu(self.fc1(context))
        return self.fc2(x)


# ----- ResNet50 + LSTM -----
class ResNet50_LSTM(nn.Module):
    """ResNet50 com a backbone + LSTM (2 capes, hidden 256)."""

    def __init__(self, hidden_size=256):
        super().__init__()
        from torchvision.models import ResNet50_Weights
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.cnn.parameters():
            p.requires_grad = False
        for p in self.cnn[7].parameters():
            p.requires_grad = True

        fused_dim = 2048 * 3 + 6 + 1
        self.bottleneck = nn.Linear(fused_dim, 512)
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size,
                            num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, ppl, xpl):
        B, T, C, H, W = ppl.shape
        ppl_feat = self.cnn(ppl.reshape(B * T, C, H, W)).view(B, T, -1)
        xpl_feat = self.cnn(xpl.reshape(B * T, C, H, W)).view(B, T, -1)

        ppl_mag = torch.norm(ppl_feat, dim=2, keepdim=True)
        xpl_mag = torch.norm(xpl_feat, dim=2, keepdim=True)
        ppl_mask = (ppl_mag > ppl_mag.mean(dim=1, keepdim=True) * 0.5).float()
        xpl_mask = (xpl_mag > xpl_mag.mean(dim=1, keepdim=True) * 0.5).float()
        ppl_intensity = (ppl_feat.mean(dim=2, keepdim=True) * ppl_mask) / (ppl_mask + 1e-6)
        xpl_intensity = (xpl_feat.mean(dim=2, keepdim=True) * xpl_mask) / (xpl_mask + 1e-6)

        d_ppl = ppl_intensity[:, 4:] - ppl_intensity[:, :-4]
        d_xpl = xpl_intensity[:, 4:] - xpl_intensity[:, :-4]
        d_ppl = torch.cat([d_ppl[:, 0:1], d_ppl[:, 0:1], d_ppl,
                           d_ppl[:, -1:], d_ppl[:, -1:]], dim=1)
        d_xpl = torch.cat([d_xpl[:, 0:1], d_xpl[:, 0:1], d_xpl,
                           d_xpl[:, -1:], d_xpl[:, -1:]], dim=1)
        d2_xpl = d_xpl[:, 4:] - d_xpl[:, :-4]
        d2_xpl = torch.cat([d2_xpl[:, 0:1], d2_xpl[:, 0:1], d2_xpl,
                            d2_xpl[:, -1:], d2_xpl[:, -1:]], dim=1)

        diff_feat = torch.abs(ppl_feat - xpl_feat)
        features = torch.cat([
            ppl_feat, xpl_feat, diff_feat, ppl_intensity, xpl_intensity,
            d_ppl, d_xpl, d2_xpl
        ], dim=2)

        angles = torch.arange(T, device=features.device) * (2 * np.pi / T)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(2).repeat(B, 1, 1)
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(2).repeat(B, 1, 1)
        features = torch.cat([features, sin, cos], dim=2)

        features = torch.relu(self.bottleneck(features))
        lstm_out, _ = self.lstm(features)
        context = lstm_out[:, -1, :]
        x = torch.relu(self.fc1(context))
        x = self.dropout(x)
        return self.fc2(x)


# ----- ResNet18 + MeanPooling -----
class ResNet18_MeanPooling(nn.Module):
    """ResNet18 + mean pooling temporal — baseline simple sense atenció ni LSTM."""

    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet18_Weights
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.cnn.parameters():
            p.requires_grad = False
        for p in self.cnn[7].parameters():
            p.requires_grad = True

        fused_dim = 512 * 3 + 5 + 2
        self.bottleneck = nn.Linear(fused_dim, 512)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, ppl, xpl):
        B, T, C, H, W = ppl.shape
        ppl_feat = self.cnn(ppl.reshape(B * T, C, H, W)).view(B, T, -1)
        xpl_feat = self.cnn(xpl.reshape(B * T, C, H, W)).view(B, T, -1)

        ppl_mag = torch.norm(ppl_feat, dim=2, keepdim=True)
        xpl_mag = torch.norm(xpl_feat, dim=2, keepdim=True)
        ppl_mask = (ppl_mag > ppl_mag.mean(dim=1, keepdim=True) * 0.5).float()
        xpl_mask = (xpl_mag > xpl_mag.mean(dim=1, keepdim=True) * 0.5).float()
        ppl_intensity = (ppl_feat.mean(dim=2, keepdim=True) * ppl_mask) / (ppl_mask + 1e-6)
        xpl_intensity = (xpl_feat.mean(dim=2, keepdim=True) * xpl_mask) / (xpl_mask + 1e-6)

        delta_ppl = ppl_intensity[:, 4:] - ppl_intensity[:, :-4]
        delta_xpl = xpl_intensity[:, 4:] - xpl_intensity[:, :-4]
        delta_ppl = torch.cat([delta_ppl[:, -2:-1], delta_ppl[:, -1:], delta_ppl,
                               delta_ppl[:, 0:1], delta_ppl[:, 1:2]], dim=1)
        delta_xpl = torch.cat([delta_xpl[:, -2:-1], delta_xpl[:, -1:], delta_xpl,
                               delta_xpl[:, 0:1], delta_xpl[:, 1:2]], dim=1)
        delta2_xpl = delta_xpl[:, 4:] - delta_xpl[:, :-4]
        delta2_xpl = torch.cat([delta2_xpl[:, -2:-1], delta2_xpl[:, -1:], delta2_xpl,
                                delta2_xpl[:, 0:1], delta2_xpl[:, 1:2]], dim=1)

        diff_feat = torch.abs(ppl_feat - xpl_feat)
        angles = torch.arange(T, device=ppl_feat.device) * (2 * np.pi / T)
        sin_enc = torch.sin(angles).view(1, T, 1).expand(B, T, 1)
        cos_enc = torch.cos(angles).view(1, T, 1).expand(B, T, 1)

        features = torch.cat([
            ppl_feat, xpl_feat, diff_feat, ppl_intensity, xpl_intensity,
            delta_ppl, delta_xpl, delta2_xpl, sin_enc, cos_enc
        ], dim=2)
        features = self.bottleneck(features)
        context = torch.mean(features, dim=1)
        x = torch.relu(self.fc1(context))
        return self.fc2(x)


# ----- ResNet50 + MeanPooling -----
class ResNet50_MeanPooling(nn.Module):
    """ResNet50 + mean pooling temporal."""

    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet50_Weights
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.cnn.parameters():
            p.requires_grad = False
        for p in self.cnn[7].parameters():
            p.requires_grad = True

        fused_dim = 2048 * 3 + 7
        self.bottleneck = nn.Linear(fused_dim, 512)
        self.fc1 = nn.Linear(512, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, ppl, xpl):
        B, T, C, H, W = ppl.shape
        ppl_feat = self.cnn(ppl.reshape(B * T, C, H, W)).view(B, T, -1)
        xpl_feat = self.cnn(xpl.reshape(B * T, C, H, W)).view(B, T, -1)

        ppl_mag = torch.norm(ppl_feat, dim=2, keepdim=True)
        xpl_mag = torch.norm(xpl_feat, dim=2, keepdim=True)
        ppl_mask = (ppl_mag > ppl_mag.mean(dim=1, keepdim=True) * 0.5).float()
        xpl_mask = (xpl_mag > xpl_mag.mean(dim=1, keepdim=True) * 0.5).float()
        ppl_intensity = (ppl_feat.mean(dim=2, keepdim=True) * ppl_mask) / (ppl_mask + 1e-6)
        xpl_intensity = (xpl_feat.mean(dim=2, keepdim=True) * xpl_mask) / (xpl_mask + 1e-6)

        d_ppl = ppl_intensity[:, 4:] - ppl_intensity[:, :-4]
        d_xpl = xpl_intensity[:, 4:] - xpl_intensity[:, :-4]
        d_ppl = torch.cat([d_ppl[:, 0:1], d_ppl[:, 0:1], d_ppl,
                           d_ppl[:, -1:], d_ppl[:, -1:]], dim=1)
        d_xpl = torch.cat([d_xpl[:, 0:1], d_xpl[:, 0:1], d_xpl,
                           d_xpl[:, -1:], d_xpl[:, -1:]], dim=1)
        d2_xpl = d_xpl[:, 4:] - d_xpl[:, :-4]
        d2_xpl = torch.cat([d2_xpl[:, 0:1], d2_xpl[:, 0:1], d2_xpl,
                            d2_xpl[:, -1:], d2_xpl[:, -1:]], dim=1)

        diff_feat = torch.abs(ppl_feat - xpl_feat)
        features = torch.cat([
            ppl_feat, xpl_feat, diff_feat, ppl_intensity, xpl_intensity,
            d_ppl, d_xpl, d2_xpl
        ], dim=2)

        angles = torch.arange(T, device=features.device) * (2 * np.pi / T)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(2).repeat(B, 1, 1)
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(2).repeat(B, 1, 1)
        features = torch.cat([features, sin, cos], dim=2)

        features = torch.relu(self.bottleneck(features))
        context = torch.mean(features, dim=1)
        x = torch.relu(self.fc1(context))
        x = self.dropout(x)
        return self.fc2(x)


# ----- ResNet18 + StaticImage -----
class ResNet18_StaticImage(nn.Module):
    """ResNet18 sobre un sol frame — baseline sense informació temporal."""

    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet18_Weights
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.cnn.parameters():
            p.requires_grad = False
        for p in self.cnn[7].parameters():
            p.requires_grad = True

        fused_dim = 512 * 3
        self.fc1 = nn.Linear(fused_dim, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, ppl, xpl):
        ppl_feat = self.cnn(ppl).view(ppl.size(0), -1)
        xpl_feat = self.cnn(xpl).view(xpl.size(0), -1)
        diff_feat = torch.abs(ppl_feat - xpl_feat)
        features = torch.cat([ppl_feat, xpl_feat, diff_feat], dim=1)
        x = torch.relu(self.fc1(features))
        return self.fc2(x)


##############################################################################
# CONFIGURACIONS PER MODEL
##############################################################################

MODEL_CONFIGS = {
    "1": {
        "name": "ResNet18 + Attention",
        "class": ResNet18_Attention,
        "results_dir": "Models/resnet18_final_attention",
        "dataset_type": "video",
        "cache": True, "sort_files": True,
        "augment_mode": "gpu_video",
        "epochs": 30, "patience": 25,
        "use_scheduler": True, "use_scaler": True,
        "stop_on_perfect_f1": False,
    },
    "2": {
        "name": "ResNet50 + Attention",
        "class": ResNet50_Attention,
        "results_dir": "Models/resnet50_final_attention",
        "dataset_type": "video",
        "cache": False, "sort_files": False,
        "augment_mode": "cpu_video",
        "epochs": 30, "patience": 20,
        "use_scheduler": False, "use_scaler": False,
        "stop_on_perfect_f1": True,
    },
    "3": {
        "name": "ResNet18 + LSTM",
        "class": ResNet18_LSTM,
        "results_dir": "Models/resnet18_final_lstm",
        "dataset_type": "video",
        "cache": True, "sort_files": True,
        "augment_mode": "gpu_video",
        "epochs": 30, "patience": 25,
        "use_scheduler": True, "use_scaler": True,
        "stop_on_perfect_f1": False,
    },
    "4": {
        "name": "ResNet50 + LSTM",
        "class": ResNet50_LSTM,
        "results_dir": "Models/resnet50_final_lstm",
        "dataset_type": "video",
        "cache": False, "sort_files": False,
        "augment_mode": "cpu_video",
        "epochs": 30, "patience": 20,
        "use_scheduler": False, "use_scaler": False,
        "stop_on_perfect_f1": True,
    },
    "5": {
        "name": "ResNet18 + MeanPooling",
        "class": ResNet18_MeanPooling,
        "results_dir": "Models/resnet18_final_meanpooling",
        "dataset_type": "video",
        "cache": True, "sort_files": True,
        "augment_mode": "gpu_video",
        "epochs": 30, "patience": 25,
        "use_scheduler": True, "use_scaler": True,
        "stop_on_perfect_f1": False,
    },
    "6": {
        "name": "ResNet50 + MeanPooling",
        "class": ResNet50_MeanPooling,
        "results_dir": "Models/resnet50_final_meanpooling",
        "dataset_type": "video",
        "cache": False, "sort_files": False,
        "augment_mode": "cpu_video",
        "epochs": 30, "patience": 20,
        "use_scheduler": False, "use_scaler": False,
        "stop_on_perfect_f1": True,
    },
    "7": {
        "name": "ResNet18 + StaticImage",
        "class": ResNet18_StaticImage,
        "results_dir": "Models/resnet18_final_static_image",
        "dataset_type": "static",
        "cache": True, "sort_files": True,
        "augment_mode": "gpu_static",
        "epochs": 20, "patience": 25,
        "use_scheduler": True, "use_scaler": True,
        "stop_on_perfect_f1": False,
    },
}


##############################################################################
# BUCLE D'ENTRENAMENT
##############################################################################

def build_datasets(cfg):
    """Construeix els datasets (train/test) segons la configuració del model."""
    if cfg["dataset_type"] == "video":
        train_ds = MineralDataset(
            DATASET_PATH, is_train=True,
            cache=cfg["cache"],
            augment_in_getitem=(cfg["augment_mode"] == "cpu_video"),
            sort_files=cfg["sort_files"],
        )
        test_ds = MineralDataset(
            DATASET_PATH, is_train=False,
            cache=cfg["cache"],
            augment_in_getitem=False,
            sort_files=cfg["sort_files"],
        )
    else:
        train_ds = MineralDatasetSingleFrame(DATASET_PATH, is_train=True)
        test_ds = MineralDatasetSingleFrame(DATASET_PATH, is_train=False)
    return train_ds, test_ds


def apply_loop_augmentation(cfg, ppl, xpl):
    """Aplica l'augmentació al loop si correspon (GPU)."""
    if cfg["augment_mode"] == "gpu_video":
        return augment_video_gpu(ppl, xpl)
    if cfg["augment_mode"] == "gpu_static":
        return augment_static_gpu(ppl, xpl)
    return ppl, xpl


def train_model(cfg):
    """Entrena un model amb 5-fold StratifiedKFold i guarda el millor checkpoint per fold.

    Per cada fold construeix el model definit a `cfg["class"]`, l'entrena durant
    `cfg["epochs"]` èpoques aplicant l'augmentació indicada i guarda els pesos amb
    millor F1-macro a `cfg["results_dir"]/fold_<N>/best_model.pth`. Compatible amb
    `evaluate_models.py`, que llegeix aquests checkpoints.

    Args:
        cfg: Diccionari de configuració amb les claus:
            name (str): Nom del model (per a logs).
            class (Type[nn.Module]): Classe del model a entrenar.
            results_dir (str): Carpeta de sortida dels checkpoints.
            dataset_type (str): "video" (5D) o "static" (4D).
            cache (bool): Si precarregar els tensors a RAM.
            sort_files (bool): Si ordenar `os.listdir()` alfabèticament.
            augment_mode (str): "cpu_video", "gpu_video" o "gpu_static".
            epochs (int): Nombre màxim d'èpoques per fold.
            patience (int): Èpoques sense millora abans de l'early stopping.
            use_scheduler (bool): Activa CosineAnnealingLR.
            use_scaler (bool): Activa mixed-precision (només a CUDA).
            stop_on_perfect_f1 (bool): Atura el fold si F1=1.0.

    Returns:
        None. Els resultats es desen a disc i s'imprimeixen per consola.
    """
    print(f"\n{'#'*72}\n# ENTRENANT: {cfg['name']}\n{'#'*72}")
    print(f" Dispositiu : {DEVICE}")
    print(f" Sortida    : {cfg['results_dir']}")
    os.makedirs(cfg["results_dir"], exist_ok=True)

    dataset_train, dataset_test = build_datasets(cfg)
    labels = dataset_train.labels

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    total_folds = 5
    EPOCHS = cfg["epochs"]
    PATIENCE = cfg["patience"]
    global_start = time.time()

    # Resum de paràmetres del model
    temp = cfg["class"]()
    total_p = sum(p.numel() for p in temp.parameters())
    train_p = sum(p.numel() for p in temp.parameters() if p.requires_grad)
    print("=" * 50)
    print(f"RESUM DE PARÀMETRES — {cfg['name']}")
    print(f"Total:        {total_p:,}")
    print(f"Congelats:    {total_p - train_p:,}")
    print(f"Entrenables:  {train_p:,}")
    print("=" * 50)
    del temp

    scaler = torch.amp.GradScaler('cuda') if (cfg["use_scaler"] and DEVICE == "cuda") else None

    cv_acc, cv_f1, cv_prec, cv_rec, cv_cm = [], [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(range(len(dataset_train)), labels)):
        print(f"\n{'='*30}\n FOLD {fold + 1}/{total_folds}\n{'='*30}")
        fold_dir = os.path.join(cfg["results_dir"], f"fold_{fold + 1}")
        os.makedirs(fold_dir, exist_ok=True)
        fold_start = time.time()

        train_loader = DataLoader(
            Subset(dataset_train, train_idx), batch_size=BATCH_SIZE,
            shuffle=True, num_workers=1, pin_memory=False,
        )
        test_loader = DataLoader(
            Subset(dataset_test, test_idx), batch_size=BATCH_SIZE,
            shuffle=False, num_workers=1, pin_memory=False,
        )

        model = cfg["class"]().to(DEVICE)

        # Balanceig de classes [Biotite=61, Hornblende=36]
        class_counts = torch.tensor([61, 36], dtype=torch.float32)
        w = (1 / class_counts)
        w = (w / w.sum()).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=w)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
                     if cfg["use_scheduler"] else None)

        best_f1 = 0.0
        best_metrics = {"acc": 0, "prec": 0, "rec": 0, "cm": None}
        no_improve = 0

        for epoch in range(EPOCHS):
            ep_start = time.time()
            model.train()

            for ppl, xpl, label in train_loader:
                ppl = ppl.to(DEVICE, non_blocking=True)
                xpl = xpl.to(DEVICE, non_blocking=True)
                label = label.to(DEVICE, non_blocking=True)

                ppl, xpl = apply_loop_augmentation(cfg, ppl, xpl)

                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    with torch.amp.autocast('cuda'):
                        loss = criterion(model(ppl, xpl), label)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = criterion(model(ppl, xpl), label)
                    loss.backward()
                    optimizer.step()

            if scheduler:
                scheduler.step()

            # Avaluació
            model.eval()
            preds_all, labels_all = [], []
            with torch.no_grad():
                for ppl, xpl, label in test_loader:
                    ppl = ppl.to(DEVICE, non_blocking=True)
                    xpl = xpl.to(DEVICE, non_blocking=True)
                    if scaler:
                        with torch.amp.autocast('cuda'):
                            out = model(ppl, xpl)
                    else:
                        out = model(ppl, xpl)
                    preds_all.extend(torch.argmax(out, dim=1).cpu().numpy())
                    labels_all.extend(label.numpy())

            acc = accuracy_score(labels_all, preds_all)
            prec = precision_score(labels_all, preds_all, average="macro", zero_division=0)
            rec = recall_score(labels_all, preds_all, average="macro", zero_division=0)
            f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
            cm = confusion_matrix(labels_all, preds_all)

            ep_time = time.time() - ep_start
            elapsed = time.time() - global_start
            fold_el = time.time() - fold_start
            rem = 1.0 - (fold * EPOCHS + epoch + 1) / (total_folds * EPOCHS)
            fold_rem = 1.0 - (epoch + 1) / EPOCHS
            total_eta = elapsed / (1 / rem - 1) if rem > 1e-6 else 0.0
            fold_eta = fold_el / (1 / fold_rem - 1) if fold_rem > 1e-6 else 0.0
            lr_str = f" | LR {scheduler.get_last_lr()[0]:.2e}" if scheduler else ""

            print(f"Època {epoch + 1:02d}/{EPOCHS} | "
                  f"Acc {acc:.3f} | F1 {f1:.3f} | Prec {prec:.3f} | Rec {rec:.3f}"
                  f"{lr_str} | Temps {ep_time:.1f}s | "
                  f"ETA Fold {format_time(fold_eta)} | ETA Total {format_time(total_eta)}")

            if f1 > best_f1:
                best_f1 = f1
                best_metrics.update({"acc": acc, "prec": prec, "rec": rec, "cm": cm})
                no_improve = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "fold": fold + 1, "epoch": epoch + 1,
                    "f1": f1, "precision": prec, "recall": rec, "accuracy": acc,
                    "confusion_matrix": cm.tolist(),
                }, os.path.join(fold_dir, "best_model.pth"))
                print(f"   Millor model desat (F1={f1:.3f})")
            else:
                no_improve += 1

            # Parada anticipada per F1=1.0 (només models ResNet50)
            if cfg["stop_on_perfect_f1"] and f1 >= 1.0:
                print("   F1 perfecte assolit — aturant aquest fold.")
                break
            if no_improve >= PATIENCE:
                print(f"   Early stopping ({PATIENCE} èpoques sense millora)")
                break

        print(f"Millor F1 del Fold {fold + 1}: {best_f1:.3f}")
        cv_f1.append(best_f1)
        cv_acc.append(best_metrics["acc"])
        cv_prec.append(best_metrics["prec"])
        cv_rec.append(best_metrics["rec"])
        cv_cm.append(best_metrics["cm"])

    # Resum global
    print("\n" + "=" * 50)
    print(f" RESULTATS FINALS 5-FOLD CV — {cfg['name']}")
    print("=" * 50)
    print(f"Accuracy:  {np.mean(cv_acc):.3f} ± {np.std(cv_acc):.3f}")
    print(f"F1-Score:  {np.mean(cv_f1):.3f} ± {np.std(cv_f1):.3f}")
    print(f"Precision: {np.mean(cv_prec):.3f} ± {np.std(cv_prec):.3f}")
    print(f"Recall:    {np.mean(cv_rec):.3f} ± {np.std(cv_rec):.3f}")
    print("-" * 50)
    print(" Matriu de Confusió Global (suma 5 folds):")
    print(np.sum(cv_cm, axis=0))
    print("=" * 50)


##############################################################################
# MENÚ INTERACTIU
##############################################################################

def print_menu():
    print("\n" + "=" * 60)
    print(" MODELS DEL TFG ")
    print("=" * 60)
    for k, v in MODEL_CONFIGS.items():
        print(f"  [{k}] {v['name']}")
    print("  [0] Sortir")
    print("=" * 60)


def main():
    print(f"\n Dispositiu detectat: {DEVICE}")
    print(f" Dataset: {os.path.abspath(DATASET_PATH)}")

    while True:
        print_menu()
        choice = input(" Tria un model: ").strip()
        if choice == "0":
            print(" Fi")
            break
        if choice not in MODEL_CONFIGS:
            print(" Opció no vàlida.")
            continue
        train_model(MODEL_CONFIGS[choice])


if __name__ == "__main__":
    main()
