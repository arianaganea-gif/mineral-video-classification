##############################################################################################################

# © 2026, Ariana Ganea
# Aquesta obra està subjecta a una llicència de Creative Commons Reconeixement-NoComercial-SenseObraDerivada
# 4.0 Internacional (CC BY-NC-ND 4.0). Veure : http://creativecommons.org/licenses/by-nc-nd/4.0/

##############################################################################################################

# SCRIPT D'AVALUACIÓ I VISUALITZACIÓ DE TOTS ELS MODELS FINALS

# - Conté l'arquitectura dels 7 models entrenats finals
# - Menú interactiu per triar-ne un (o tots)
# - Avalua els 5 folds, imprimeix mètriques per consola
# - Genera visualitzacions (matrius de confusió + barres de mètriques)

##############################################################################################################

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use("Agg")

#############################################
# CONFIGURACIÓ GLOBAL
#############################################

DATASET_PATH = "Dataset/segmented_minerals_tensor"
MODELS_BASE_DIR = "Models/"
OUTPUT_DIR = "Results/"

BATCH_SIZE = 4
MAX_FRAMES = 72
CLASS_NAMES = ["Biotite", "Hornblende"]

DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available()
          else "cpu")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


##############################################################################
# DATASETS
##############################################################################

class MineralDataset(Dataset):
    """Dataset de vídeos sencers (PPL+XPL) per als 6 models temporals.

    sort_files=True ordena alfabèticament; els ResNet50 originals NO ordenaven
    i cal mantenir-ho per reproduir els splits de KFold del training.
    """

    def __init__(self, root_dir, max_frames=72, sort_files=True):
        self.samples = []
        self.labels = []
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = torch.load(self.samples[idx], weights_only=False)
        return data["ppl"], data["xpl"], torch.tensor(data["label"])


class MineralDatasetSingleFrame(Dataset):
    """Dataset de frame únic (frame 0) per al model d'imatge estàtica."""

    def __init__(self, root_dir):
        self.samples = []
        self.labels = []
        classes = {"Biotite": 0, "Hornblende": 1}
        for mineral, lbl in classes.items():
            mineral_path = os.path.join(root_dir, mineral)
            if not os.path.exists(mineral_path):
                continue
            for f in sorted(os.listdir(mineral_path)):
                if f.endswith(".pt"):
                    self.samples.append(os.path.join(mineral_path, f))
                    self.labels.append(lbl)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = torch.load(self.samples[idx], weights_only=False)
        return data["ppl"][0], data["xpl"][0], torch.tensor(data["label"])


##############################################################################
# ARQUITECTURES
##############################################################################

# ----- ResNet18 + Attention -----
class ResNet18_Attention(nn.Module):
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
        ppl_intensity = ppl_feat.mean(dim=2, keepdim=True) * ppl_mask
        xpl_intensity = xpl_feat.mean(dim=2, keepdim=True) * xpl_mask
        ppl_intensity = ppl_intensity / (ppl_mask + 1e-6)
        xpl_intensity = xpl_intensity / (xpl_mask + 1e-6)

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
# REGISTRE DE MODELS
##############################################################################

MODELS = {
    "1": {"name": "ResNet18 + Attention",
          "dir": "resnet18_final_attention",
          "class": ResNet18_Attention,  "type": "video", "sort_files": True},
    "2": {"name": "ResNet50 + Attention",
          "dir": "resnet50_final_attention",
          "class": ResNet50_Attention,  "type": "video", "sort_files": False},
    "3": {"name": "ResNet18 + LSTM",
          "dir": "resnet18_final_lstm",
          "class": ResNet18_LSTM,        "type": "video", "sort_files": True,
          "is_winner": True},
    "4": {"name": "ResNet50 + LSTM",
          "dir": "resnet50_final_lstm",
          "class": ResNet50_LSTM,        "type": "video", "sort_files": False},
    "5": {"name": "ResNet18 + MeanPooling",
          "dir": "resnet18_final_meanpooling",
          "class": ResNet18_MeanPooling, "type": "video", "sort_files": True},
    "6": {"name": "ResNet50 + MeanPooling",
          "dir": "resnet50_final_meanpooling",
          "class": ResNet50_MeanPooling, "type": "video", "sort_files": False},
    "7": {"name": "ResNet18 + StaticImage",
          "dir": "resnet18_final_static_image",
          "class": ResNet18_StaticImage, "type": "static", "sort_files": True},
}


##############################################################################
# AVALUACIÓ
##############################################################################

def build_dataset(cfg):
    if cfg["type"] == "static":
        return MineralDatasetSingleFrame(DATASET_PATH)
    return MineralDataset(DATASET_PATH, MAX_FRAMES, sort_files=cfg.get("sort_files", True))


def evaluate_model(cfg):
    """Avalua un model entrenat sobre els 5 folds del StratifiedKFold.

    Reprodueix els mateixos splits que el training (`random_state=42`), carrega
    el `best_model.pth` de cada fold, calcula Accuracy, Precision, Recall i F1
    (macro) i recull les probabilitats i errors per a les visualitzacions
    posteriors. Si un fold no té checkpoint, el salta i continua.

    Args:
        cfg: Diccionari de configuració amb les claus:
            name (str): Nom del model (per a logs).
            dir (str): Carpeta dels checkpoints (relativa a `MODELS_BASE_DIR`).
            class (Type[nn.Module]): Classe del model a instanciar.
            type (str): "video" (5D) o "static" (4D).
            is_winner (bool, opcional): Si és True activa visualitzacions extra.

    Returns:
        dict | None: Diccionari amb mètriques per fold, mitjanes ± std, matrius
        de confusió, probabilitats predites, errors detallats i el dataset usat.
        Retorna None si no s'ha trobat el directori del model.
    """
    model_dir = os.path.join(MODELS_BASE_DIR, cfg["dir"])
    if not os.path.exists(model_dir):
        print(f" No s'ha trobat el directori del model: {model_dir}")
        return None

    dataset = build_dataset(cfg)
    labels = dataset.labels
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    cms = []
    all_pred_probs = []     # P(classe=Hornblenda) per a cada mostra de test
    all_true_labels = []
    errors_detallats = []   # FP/FN amb fitxer i confiança (per a l'anàlisi visual)

    print(f"\n{'='*60}\n AVALUANT: {cfg['name']}\n Carpeta: {model_dir}\n{'='*60}")

    for fold, (_, test_idx) in enumerate(skf.split(range(len(dataset)), labels)):
        fold_num = fold + 1
        ckpt_path = os.path.join(model_dir, f"fold_{fold_num}", "best_model.pth")

        if not os.path.exists(ckpt_path):
            print(f"  Fold {fold_num}: model no trobat ({ckpt_path})")
            continue

        loader = DataLoader(Subset(dataset, test_idx),
                            batch_size=BATCH_SIZE, shuffle=False)

        model = cfg["class"]().to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        preds_all, labels_all = [], []
        with torch.no_grad():
            for batch_i, (ppl, xpl, lbl) in enumerate(loader):
                ppl, xpl = ppl.to(DEVICE), xpl.to(DEVICE)
                out = model(ppl, xpl)
                probs = F.softmax(out, dim=1)
                preds = torch.argmax(probs, dim=1)

                preds_all.extend(preds.cpu().numpy())
                labels_all.extend(lbl.numpy())

                all_pred_probs.extend(probs[:, 1].cpu().numpy())
                all_true_labels.extend(lbl.numpy())

                # Recollim errors per a l'anàlisi visual (només cal pel guanyador,
                # però és barat fer-ho sempre)
                for b in range(len(lbl)):
                    pred_c = preds[b].item()
                    true_c = lbl[b].item()
                    if pred_c != true_c:
                        global_idx = test_idx[batch_i * BATCH_SIZE + b]
                        errors_detallats.append({
                            "global_idx": int(global_idx),
                            "fitxer": os.path.basename(dataset.samples[global_idx]),
                            "real": true_c,
                            "pred": pred_c,
                            "nom_real": "Hornblenda" if true_c == 1 else "Biotita",
                            "nom_pred": "Hornblenda" if pred_c == 1 else "Biotita",
                            "confianca": float(probs[b, pred_c].item() * 100),
                        })

        acc = accuracy_score(labels_all, preds_all)
        prec = precision_score(labels_all, preds_all, average="macro", zero_division=0)
        rec = recall_score(labels_all, preds_all, average="macro", zero_division=0)
        f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
        cm = confusion_matrix(labels_all, preds_all, labels=[0, 1])

        metrics["accuracy"].append(acc)
        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        metrics["f1"].append(f1)
        cms.append(cm)

        print(f"\n--- FOLD {fold_num} ---")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Matriu de Confusió:\n{cm}")

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    if not metrics["accuracy"]:
        print(" Cap fold avaluat correctament.")
        return None

    print("\n" + "=" * 60)
    print(f" RESUM GLOBAL — {cfg['name']} (Mitjana ± Desviació)")
    print("=" * 60)
    summary = {}
    for k, vals in metrics.items():
        m, s = float(np.mean(vals)), float(np.std(vals))
        summary[k] = (m, s)
        print(f"  {k.capitalize():<10}: {m:.4f} ± {s:.4f}")
    cm_global = np.sum(cms, axis=0)
    print(f"\n Matriu de Confusió Global:\n{cm_global}")

    return {"name": cfg["name"], "dir": cfg["dir"],
            "metrics": metrics, "summary": summary,
            "cms": cms, "cm_global": cm_global,
            "all_pred_probs": all_pred_probs,
            "all_true_labels": all_true_labels,
            "errors": errors_detallats,
            "dataset": dataset,
            "is_winner": cfg.get("is_winner", False)}


##############################################################################
# VISUALITZACIONS
##############################################################################

def _plot_cm(ax, cm, title, normalize=False, cmap="Blues"):
    if normalize:
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        data = cm_norm
        fmt = ".2f"
    else:
        data = cm
        fmt = "d"

    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=data.max() if data.max() > 0 else 1)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicció")
    ax.set_ylabel("Real")
    ax.set_title(title, fontsize=10)

    threshold = data.max() / 2 if data.max() > 0 else 0.5
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = "white" if data[i, j] > threshold else "black"
            ax.text(j, i, format(data[i, j], fmt),
                    ha="center", va="center", color=color, fontsize=11)
    return im


def visualize_model(result, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    name = result["name"]
    cms = result["cms"]
    cm_global = result["cm_global"]
    metrics = result["metrics"]
    summary = result["summary"]

    # ---------- 1) Matrius de confusió per fold + global ----------
    n_folds = len(cms)
    fig, axes = plt.subplots(2, max(3, n_folds), figsize=(4 * max(3, n_folds), 8))
    fig.suptitle(f"Matrius de Confusió — {name}", fontsize=14, fontweight="bold")

    for i, cm in enumerate(cms):
        _plot_cm(axes[0, i], cm, f"Fold {i+1} (absoluta)")
        _plot_cm(axes[1, i], cm, f"Fold {i+1} (normalitzada)", normalize=True, cmap="Greens")

    # Apaga eixos sobrants
    for j in range(n_folds, axes.shape[1]):
        axes[0, j].axis("off")
        axes[1, j].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    cm_path = os.path.join(output_dir, "matrius_confusio_per_fold.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   {cm_path}")

    # ---------- 2) Matriu de confusió global ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"Matriu de Confusió Global (suma 5 folds) — {name}",
                 fontsize=13, fontweight="bold")
    _plot_cm(axes[0], cm_global, "Absoluta")
    _plot_cm(axes[1], cm_global, "Normalitzada", normalize=True, cmap="Greens")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    cmg_path = os.path.join(output_dir, "matriu_confusio_global.png")
    plt.savefig(cmg_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   {cmg_path}")

    # ---------- 3) Mètriques per fold (gràfic de línies) ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    folds = np.arange(1, n_folds + 1)
    for k, color in zip(["accuracy", "f1", "precision", "recall"],
                        ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]):
        ax.plot(folds, metrics[k], "o-", label=k.capitalize(),
                color=color, linewidth=2, markersize=8)
    ax.set_xticks(folds)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Valor")
    ax.set_title(f"Mètriques per Fold — {name}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    line_path = os.path.join(output_dir, "metriques_per_fold.png")
    plt.savefig(line_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   {line_path}")

    # ---------- 4) Resum global (barres + error) ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    keys = ["accuracy", "f1", "precision", "recall"]
    means = [summary[k][0] for k in keys]
    stds = [summary[k][1] for k in keys]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    bars = ax.bar([k.capitalize() for k in keys], means, yerr=stds,
                  capsize=8, color=colors, edgecolor="black", alpha=0.85)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.01,
                f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Valor")
    ax.set_title(f"Resultats Globals 5-Fold CV — {name}",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "resum_global.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   {bar_path}")


##############################################################################
# GRÀFIQUES MEMÒRIA TFG (estil ResNet18 / ResNet50)
##############################################################################

# Etiquetes dels models tal com apareixen al gràfic de la memòria
_TFG_LABEL = {
    "resnet18_final_meanpooling":   "MeanPooling",
    "resnet18_final_attention":     "Attention",
    "resnet18_final_lstm":          "LSTM",
    "resnet18_final_static_image":  "Imatge Estàtica",
    "resnet50_final_meanpooling":   "MeanPooling",
    "resnet50_final_attention":     "Attention",
    "resnet50_final_lstm":          "LSTM",
}

# Ordre desitjat per a cada backbone
_TFG_ORDER = {
    "resnet18": ["MeanPooling", "Attention", "LSTM", "Imatge Estàtica"],
    "resnet50": ["MeanPooling", "Attention", "LSTM"],
}


def visualize_backbone_tfg(all_results, backbone, output_dir):
    """Reprodueix les figures per a la memòria del TFG.

    backbone: "resnet18" o "resnet50".
    Utilitza els valors CALCULATS (mitjana ± std de cada mètrica).
    """
    # Filtrem només els resultats del backbone demanat
    label_by_dir = {d: l for d, l in _TFG_LABEL.items() if d.startswith(backbone)}
    by_label = {}
    for r in all_results:
        if r["dir"] in label_by_dir:
            by_label[label_by_dir[r["dir"]]] = r

    order = [m for m in _TFG_ORDER[backbone] if m in by_label]
    if not order:
        return

    metrics_labels = ["Precisió global", "Puntuació F1", "Precisió", "Sensibilitat"]
    metric_keys = ["accuracy", "f1", "precision", "recall"]

    means = {m: [by_label[m]["summary"][k][0] for k in metric_keys] for m in order}
    stds  = {m: [by_label[m]["summary"][k][1] for k in metric_keys] for m in order}

    colors_palette = ['#D9D9D9', '#9ECAE1', '#4292C6', '#084594']
    colors = colors_palette[:len(order)]

    x = np.arange(len(metrics_labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('white')

    for i, (model, color) in enumerate(zip(order, colors)):
        offset = width * i
        ax.bar(x + offset, means[model], width, yerr=stds[model], label=model,
               color=color, edgecolor='white', linewidth=1.5,
               capsize=5, ecolor='#555555', error_kw={'elinewidth': 1.5})

    ax.set_ylabel('Rendiment', fontsize=12, fontweight='bold',
                  color='#333333', labelpad=15)
    pretty = "ResNet18" if backbone == "resnet18" else "ResNet50"
    ax.set_title(f'Comparativa dels Models {pretty}', fontsize=16,
                 fontweight='bold', color='#333333', pad=25)

    ax.set_xticks(x + width * (len(order) - 1) / 2)
    ax.set_xticklabels(metrics_labels, fontsize=12, fontweight='bold', color='#333333')
    ax.tick_params(axis='x', bottom=False)

    # Eix Y adaptatiu
    if backbone == "resnet18":
        ax.set_ylim(0.65, 1.0)
    else:
        ax.set_ylim(0.70, 1.1)
    ax.tick_params(axis='y', colors='#555555')

    ax.yaxis.grid(True, linestyle='-', which='major', color='#EEEEEE', alpha=1.0)
    ax.set_axisbelow(True)

    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.spines['bottom'].set_linewidth(1.5)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
              ncol=len(order), frameon=False, fontsize=11)

    plt.tight_layout()
    out = os.path.join(output_dir, f"comparativa_{backbone}_TFG.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f" Gràfica TFG {pretty} desada: {out}")


def visualize_comparison(all_results, output_dir):
    """Gràfic comparatiu entre tots els models avaluats."""
    if len(all_results) < 2:
        return
    os.makedirs(output_dir, exist_ok=True)

    names = [r["name"] for r in all_results]
    keys = ["accuracy", "f1", "precision", "recall"]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(names)), 6))
    for i, (k, c) in enumerate(zip(keys, colors)):
        means = [r["summary"][k][0] for r in all_results]
        stds = [r["summary"][k][1] for r in all_results]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, capsize=4,
               label=k.capitalize(), color=c, edgecolor="black", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Valor")
    ax.set_title("Comparativa entre models (5-Fold CV)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    out = os.path.join(output_dir, "comparativa_models.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n Comparativa global desada: {out}")

    # Taula de text resum
    print("\n" + "=" * 78)
    print(" TAULA RESUM (Mitjana ± Desviació)")
    print("=" * 78)
    print(f"{'Model':<28}{'Accuracy':<14}{'F1':<14}{'Precision':<14}{'Recall':<14}")
    print("-" * 78)
    for r in all_results:
        s = r["summary"]
        line = f"{r['name']:<28}"
        for k in keys:
            line += f"{s[k][0]:.3f}±{s[k][1]:.3f}  "
        print(line)
    print("=" * 78)


##############################################################################
# VISUALITZACIONS EXTRA (model guanyador)
##############################################################################

def _white_bg(img, threshold=0.20):
    """Substitueix el fons uniforme (detectat per color de cantonades) per blanc."""
    corners = np.stack([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]])
    bg = corners.mean(axis=0)
    mask = np.linalg.norm(img - bg, axis=2) < threshold
    out = img.copy()
    out[mask] = 1.0
    return out


def _tight_crop_white(img, pad=6, white_thr=0.97):
    """Retalla al quadrat mínim que envolta els píxels no-blancs (el mineral)."""
    gray = img.mean(axis=2)
    non_white = gray < white_thr
    if not non_white.any():
        return img
    rows = np.where(non_white.any(axis=1))[0]
    cols = np.where(non_white.any(axis=0))[0]
    H, W = img.shape[:2]
    r0 = max(0, rows[0] - pad)
    r1 = min(H, rows[-1] + pad + 1)
    c0 = max(0, cols[0] - pad)
    c1 = min(W, cols[-1] + pad + 1)
    h, w = r1 - r0, c1 - c0
    side = max(h, w)
    if h < side:
        extra = side - h
        r0 = max(0, r0 - extra // 2)
        r1 = min(H, r0 + side)
        r0 = max(0, r1 - side)
    if w < side:
        extra = side - w
        c0 = max(0, c0 - extra // 2)
        c1 = min(W, c0 + side)
        c0 = max(0, c1 - side)
    return img[r0:r1, c0:c1]


def _crop_to_mineral(ppl_tensor, xpl_tensor, pad=10, std_thr=0.015):
    """Retalla ambdós tensors (T,C,H,W) al quadrat mínim que envolta el
    mineral, detectat per variabilitat temporal al PPL. Retorna quadrat."""
    _mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    gray  = (ppl_tensor.float() * _std + _mean).clamp(0, 1).mean(dim=1)
    std_map = gray.std(dim=0)
    mask = std_map > std_thr
    if mask.sum() < 4:
        return ppl_tensor, xpl_tensor
    rows = torch.where(mask.any(dim=1))[0]
    cols = torch.where(mask.any(dim=0))[0]
    H, W = ppl_tensor.shape[2], ppl_tensor.shape[3]
    r0 = max(0, rows[0].item() - pad)
    r1 = min(H, rows[-1].item() + pad + 1)
    c0 = max(0, cols[0].item() - pad)
    c1 = min(W, cols[-1].item() + pad + 1)
    # Fem quadrat (dimensió màxima)
    h, w = r1 - r0, c1 - c0
    side  = max(h, w)
    if h < side:
        extra = side - h
        r0 = max(0, r0 - extra // 2)
        r1 = min(H, r0 + side)
        r0 = max(0, r1 - side)
    if w < side:
        extra = side - w
        c0 = max(0, c0 - extra // 2)
        c1 = min(W, c0 + side)
        c0 = max(0, c1 - side)
    return ppl_tensor[:, :, r0:r1, c0:c1], xpl_tensor[:, :, r0:r1, c0:c1]


def tensor_to_img(tensor):
    """Converteix un tensor (C,H,W) a imatge numpy (H,W,C) normalitzada [0,1]."""
    img = tensor.cpu().detach().numpy().transpose(1, 2, 0)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    return img


def _tensor_to_img_real(tensor):
    """Tensor (C,H,W) normalitzat ImageNet → imatge RGB (H,W,3) amb els
    COLORS ORIGINALS denormalitzats (sense min-max scaling)."""
    _mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denorm = (tensor.float().cpu() * _std + _mean).clamp(0, 1)
    return denorm.numpy().transpose(1, 2, 0)


def _consistent_bbox(imgs, pad=8, white_thr=0.97):
    """Bbox quadrada UNIÓ dels píxels no-blancs de totes les imatges (H,W,3).
    Garanteix que tots els fotogrames quedin retallats al mateix marc."""
    H, W = imgs[0].shape[:2]
    union = np.zeros((H, W), dtype=bool)
    for im in imgs:
        union |= (im.mean(axis=2) < white_thr)
    if not union.any():
        return 0, H, 0, W
    rows = np.where(union.any(axis=1))[0]
    cols = np.where(union.any(axis=0))[0]
    r0 = max(0, rows[0] - pad)
    r1 = min(H, rows[-1] + pad + 1)
    c0 = max(0, cols[0] - pad)
    c1 = min(W, cols[-1] + pad + 1)
    h, w = r1 - r0, c1 - c0
    side = max(h, w)
    if h < side:
        extra = side - h
        r0 = max(0, r0 - extra // 2)
        r1 = min(H, r0 + side)
        r0 = max(0, r1 - side)
    if w < side:
        extra = side - w
        c0 = max(0, c0 - extra // 2)
        c1 = min(W, c0 + side)
        c0 = max(0, c1 - side)
    return r0, r1, c0, c1


def visualize_winner_extras(result, output_dir):
    """Genera les figures detallades del model guanyador (ROC, confiança,
    matriu seaborn, anàlisi visual d'errors)."""
    os.makedirs(output_dir, exist_ok=True)
    name = result["name"]
    dataset = result["dataset"]
    all_true = np.array(result["all_true_labels"])
    all_probs = np.array(result["all_pred_probs"])
    cm_global = result["cm_global"]
    errors = result["errors"]

    # 1) Corba ROC
    fpr, tpr, _ = roc_curve(all_true, all_probs)
    plt.figure(figsize=(8, 6), facecolor='white')
    plt.plot(fpr, tpr, color='#3182BD', lw=2,
             label=f'Corba ROC (AUC = {auc(fpr, tpr):.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    plt.xlabel('Taxa de falsos positius (FPR)', fontweight='bold')
    plt.ylabel('Taxa de veritables positius (TPR)', fontweight='bold')
    plt.title(f'Corba ROC ({name})', fontweight='bold', pad=15)
    plt.legend(loc="lower right", frameon=False)
    plt.grid(True, linestyle='--', alpha=0.5)
    out = os.path.join(output_dir, "corba_roc.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   {out}")

    # 2) Histograma de confiança
    probs_biotita = all_probs[all_true == 0]
    probs_hornblenda = all_probs[all_true == 1]
    plt.figure(figsize=(9, 6), facecolor='white')
    plt.hist(probs_biotita, bins=np.linspace(0, 1, 25), alpha=0.7,
             color='#E6550D', label='Biotita real', edgecolor='white')
    plt.hist(probs_hornblenda, bins=np.linspace(0, 1, 25), alpha=0.7,
             color='#3182BD', label='Hornblenda real', edgecolor='white')
    plt.axvline(0.5, color='black', linestyle='dashed', linewidth=1.5,
                label='Llindar (0.5)')
    plt.xlabel('Probabilitat de hornblenda', fontweight='bold')
    plt.ylabel('Nombre de mostres', fontweight='bold')
    plt.title('Distribució de confiança', fontweight='bold', pad=15)
    plt.legend(loc='upper right', frameon=False)
    out = os.path.join(output_dir, "distribucio_confianca.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   {out}")

    # 3) Matriu de confusió heatmap (seaborn)
    plt.figure(figsize=(7, 6), facecolor='white')
    sns.heatmap(cm_global, annot=True, fmt='d', cmap='Blues', cbar=False,
                annot_kws={"size": 16, "weight": "bold"},
                xticklabels=['Biotita', 'Hornblenda'],
                yticklabels=['Biotita', 'Hornblenda'])
    plt.xlabel('Classe predita', fontweight='bold', labelpad=10)
    plt.ylabel('Classe real', fontweight='bold', labelpad=10)
    plt.title('Matriu de confusió (5 Folds)', fontweight='bold', pad=15)
    out = os.path.join(output_dir, "matriu_confusio_heatmap.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   {out}")

    # 4) Anàlisi visual d'errors — retall al mineral, fons blanc
    fp = sorted([e for e in errors if e['real'] == 0 and e['pred'] == 1],
                key=lambda x: x['confianca'], reverse=True)
    fn = sorted([e for e in errors if e['real'] == 1 and e['pred'] == 0],
                key=lambda x: x['confianca'], reverse=True)

    # ── Llista completa per consola ──────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  FALSOS POSITIUS ({len(fp)})  — Biotita predita com hornblenda")
    print(f"{'─' * 60}")
    for i, e in enumerate(fp, 1):
        print(f"  {i:>3}. {e['fitxer']:<40}  conf={e['confianca']:5.1f}%")
        print(f"       {dataset.samples[e['global_idx']]}")
    print(f"\n{'─' * 60}")
    print(f"  FALSOS NEGATIUS ({len(fn)})  — Hornblenda predita com biotita")
    print(f"{'─' * 60}")
    for i, e in enumerate(fn, 1):
        print(f"  {i:>3}. {e['fitxer']:<40}  conf={e['confianca']:5.1f}%")
        print(f"       {dataset.samples[e['global_idx']]}")
    print(f"{'─' * 60}\n")

    def top_bottom(lst):
        if len(lst) <= 5:
            return lst
        return lst[:2] + lst[-2:]

    fp_show = top_bottom(fp)
    fn_show = top_bottom(fn)
    n_rows  = max(len(fp_show), len(fn_show), 1)

    fig, axes = plt.subplots(n_rows, 4, figsize=(14, n_rows * 4),
                             facecolor='white', squeeze=False)
    fig.suptitle("Anàlisi visual d'errors — mineral retallat, fons blanc",
                 fontsize=15, fontweight='bold', y=0.98)
    fig.text(0.25, 0.94, 'Falsos positius : real biotita → predicció hornblenda',
             ha='center', fontsize=14, fontweight='bold', color='#E6550D')
    fig.text(0.75, 0.94, 'Falsos negatius : real hornblenda → predicció biotita',
             ha='center', fontsize=14, fontweight='bold', color='#3182BD')

    pending_titles = []   # (row, col_off, text, color) → added after tight_layout

    for row in range(n_rows):
        for col_off, show_list, color in [(0, fp_show, 'black'),
                                          (2, fn_show, 'black')]:
            if row < len(show_list):
                err = show_list[row]
                ppl, xpl, _ = dataset[err['global_idx']]

                # Retall al mineral + fons blanc
                img_ppl = _tight_crop_white(_white_bg(tensor_to_img(ppl[30])))
                img_xpl = _tight_crop_white(_white_bg(tensor_to_img(xpl[30])))

                axes[row, col_off].imshow(img_ppl, aspect='auto', interpolation='bilinear')
                axes[row, col_off + 1].imshow(img_xpl, aspect='auto', interpolation='bilinear')

                # Badge PPL / XPL sobre la imatge
                #3
                #_badge = dict(boxstyle='round,pad=0.22', facecolor='#1A1A1A',
                #              alpha=0.62, edgecolor='none')
                #for ax, lbl in [(axes[row, col_off], 'PPL'),
                #                (axes[row, col_off + 1], 'XPL')]:
                #    ax.text(0.04, 0.05, lbl, transform=ax.transAxes,
                #            ha='left', va='bottom', fontsize=10,
                #            fontweight='bold', color='white', bbox=_badge)

                label = "Molt segur" if row < 2 else "Poc segur (dubte)"
                pending_titles.append((
                    row, col_off,
                    f"Confiança: {err['confianca']:.1f}%  ·  {label}\n{err['fitxer']}",
                    'black'))
            else:
                axes[row, col_off].axis('off')
                axes[row, col_off + 1].axis('off')

            for c in [col_off, col_off + 1]:
                axes[row, c].set_xticks([])
                axes[row, c].set_yticks([])
                for sp in axes[row, c].spines.values():
                    sp.set_visible(False)

    fig.add_artist(plt.Line2D((0.5, 0.5), (0.05, 0.92),
                              color='#CCC', linewidth=1.2, linestyle='--'))
    plt.tight_layout(rect=[0, 0.03, 1, 0.92], h_pad=3.5, w_pad=1.5)

    # Títols centrats entre PPL i XPL — calculats DESPRÉS de tight_layout
    for row, col_off, text, color in pending_titles:
        ax_ppl = axes[row, col_off]
        ax_xpl = axes[row, col_off + 1]
        bb_ppl = ax_ppl.get_position()
        bb_xpl = ax_xpl.get_position()
        x_mid = (bb_ppl.x0 + bb_xpl.x1) / 2
        y_top = (bb_ppl.y1 + bb_xpl.y1) / 2
        fig.text(x_mid, y_top + 0.004, text,
                 ha='center', va='bottom',
                 fontsize=12, color='black',
                 transform=fig.transFigure)
    out = os.path.join(output_dir, "analisi_errors_visual.png")
    plt.savefig(out, dpi=300, facecolor='white',
                transparent=True, bbox_inches='tight')
    plt.close()
    print(f"   {out}")



##############################################################################
# VISUALITZACIÓ ROTACIONS PER ERROR — colors reals, 0°→300° cada 60°
##############################################################################

def visualize_winner_rotations(result, output_dir):
    """Per cada fals positiu i negatiu del model guanyador, genera una figura
    2×6 (PPL/XPL × 0°,60°,120°,180°,240°,300°) amb els COLORS ORIGINALS
    del mineral segmentat, fons blanc i retall ajustat."""
    out_dir = os.path.join(output_dir, "rotations_errors")
    os.makedirs(out_dir, exist_ok=True)
    dataset = result["dataset"]
    errors  = result["errors"]

    fp = sorted([e for e in errors if e['real'] == 0 and e['pred'] == 1],
                key=lambda x: x['confianca'], reverse=True)
    fn = sorted([e for e in errors if e['real'] == 1 and e['pred'] == 0],
                key=lambda x: x['confianca'], reverse=True)

    angles = [0, 60, 120, 180, 240, 300]
    idxs   = [a // 5 for a in angles]   # 72 fotogrames a pas de 5° → 0,12,24,36,48,60

    def _render(err, tipus):
        ppl, xpl, _ = dataset[err['global_idx']]
        # 12 fotogrames amb colors REALS denormalitzats + fons blanc
        # Retall tight INDIVIDUAL per fotograma → cada mineral omple el seu
        # subplot (el dataset té augmentació rotacional i la posició varia)
        ppl_imgs = [_tight_crop_white(_white_bg(_tensor_to_img_real(ppl[i])),
                                      pad=4) for i in idxs]
        xpl_imgs = [_tight_crop_white(_white_bg(_tensor_to_img_real(xpl[i])),
                                      pad=4) for i in idxs]

        fig, axes = plt.subplots(2, 6, figsize=(15, 5.8),
                                 facecolor='white', squeeze=False)
        fig.suptitle(err['fitxer'],
                     fontsize=16, color='black', y=0.995)

        for col, ang in enumerate(angles):
            axes[0, col].imshow(ppl_imgs[col], aspect='auto',
                                interpolation='bilinear')
            axes[1, col].imshow(xpl_imgs[col], aspect='auto',
                                interpolation='bilinear')
            axes[0, col].set_title(f"{ang}°", fontsize=13,
                                   color='#222', pad=8)
            for r in [0, 1]:
                axes[r, col].set_xticks([])
                axes[r, col].set_yticks([])
                for sp in axes[r, col].spines.values():
                    sp.set_visible(False)

        # Etiquetes verticals PPL / XPL a la primera columna
        axes[0, 0].set_ylabel("PPL", fontsize=13,
                              color='#222', labelpad=14)
        axes[1, 0].set_ylabel("XPL", fontsize=13,
                              color='#222', labelpad=14)

        plt.tight_layout(rect=[0, 0, 1, 0.91], h_pad=1.2, w_pad=0.5)
        safe = err['fitxer'].replace('.pt', '').replace('/', '_')
        out = os.path.join(out_dir, f"{tipus}_{safe}.png")
        plt.savefig(out, dpi=200, facecolor='white', bbox_inches='tight')
        plt.close()
        print(f"   {out}")

    print(f"\n  Generant rotacions per error: {len(fp)} FP + {len(fn)} FN")
    for err in fp:
        _render(err, 'FP')
    for err in fn:
        _render(err, 'FN')



##############################################################################
# MENÚ INTERACTIU
##############################################################################

def print_menu():
    print("\n" + "=" * 60)
    print(" VISUALITZADOR UNIVERSAL DE RESULTATS — TFG MINERALS")
    print("=" * 60)
    for k, v in MODELS.items():
        winner_tag = "  (guanyador)" if v.get("is_winner") else ""
        marker = "" if os.path.exists(os.path.join(MODELS_BASE_DIR, v["dir"])) else "  [no trobat]"
        print(f"  [{k}] {v['name']}{winner_tag}{marker}")
    print(f"  [8] Avaluar TOTS els models i generar comparativa")
    print(f"  [0] Sortir")
    print("=" * 60)


def run(choice):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if choice == "8":
        all_results = []
        for k in sorted(MODELS.keys()):
            res = evaluate_model(MODELS[k])
            if res is not None:
                model_out = os.path.join(OUTPUT_DIR, MODELS[k]["dir"])
                visualize_model(res, model_out)
                if res.get("is_winner"):
                    visualize_winner_extras(res, model_out)
                    visualize_winner_rotations(res, model_out)
                all_results.append(res)
        visualize_comparison(all_results, OUTPUT_DIR)
        visualize_backbone_tfg(all_results, "resnet18", OUTPUT_DIR)
        visualize_backbone_tfg(all_results, "resnet50", OUTPUT_DIR)
        return

    if choice not in MODELS:
        print(" Opció no vàlida.")
        return

    cfg = MODELS[choice]
    res = evaluate_model(cfg)
    if res is not None:
        model_out = os.path.join(OUTPUT_DIR, cfg["dir"])
        visualize_model(res, model_out)
        if res.get("is_winner"):
            visualize_winner_extras(res, model_out)
            visualize_winner_rotations(res, model_out)


def main():
    print(f"\n Dispositiu: {DEVICE}")
    print(f" Dataset: {os.path.abspath(DATASET_PATH)}")
    print(f" Models: {os.path.abspath(MODELS_BASE_DIR)}")
    print(f" Sortida visualitzacions: {os.path.abspath(OUTPUT_DIR)}")

    while True:
        print_menu()
        choice = input(" Tria una opció: ").strip()
        if choice == "0":
            print(" Fi")
            break
        run(choice)


if __name__ == "__main__":
    main()
