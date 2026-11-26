# Classificació de Minerals — TFG

Pipeline complet per classificar **biotita** vs **hornblenda** a partir de seqüències de vídeo de microscòpia petrogràfica (PPL+XPL, gir de 360°). Inclou descàrrega de dades, segmentació amb SAM2, conversió a tensors, set de 7 models entrenats amb 5-fold CV i scripts d'avaluació/visualització.

> © 2026, Ariana Ganea — CC BY-NC-ND 4.0

---

## Estructura del projecte

```
MINERAL_CLASSIFICATION_TFG/
├── README.md                       ← aquest fitxer
├── requirements.txt
├── train_models.py                 ← entrenament dels 7 models (menú interactiu)
├── evaluate_models.py              ← avaluació + visualitzacions
├── gradcam_attention.py            ← Grad-CAM sobre el ResNet50 + Attention
│
├── Dataset/
│   ├── METADADES.md                Descripció del dataset
│   ├── download_minerals.py        Descàrrega dels frames (Biotita + Hornblenda)
│   ├── original_videos/            JPG descarregats del microscopi virtual
│   ├── segmented_minerals/         PNG dels cristalls aïllats amb SAM2
│   └── segmented_minerals_tensor/  Tensors `.pt` (consum dels models)
│
├── Segmentation/
│   ├── Codi_Segmentacion.py        Eina interactiva (SAM2 + GUI OpenCV)
│   ├── MirarSegmentacio.py         Visualitzador per inspeccionar resultats
│   ├── TF2Tensor.py                PNG → tensors `.pt`
│   └── sam2_hiera_large.pt         Pesos del model SAM2
│
├── Models/                         ← Checkpoints generats per `train_models.py`
│   ├── resnet18_final_attention/fold_{1..5}/best_model.pth
│   ├── resnet18_final_lstm/...
│   ├── resnet18_final_meanpooling/...
│   ├── resnet18_final_static_image/...
│   ├── resnet50_final_attention/...
│   ├── resnet50_final_lstm/...
│   └── resnet50_final_meanpooling/...
│
└── Results/                        ← Sortida de `evaluate_models.py` i `gradcam_attention.py`
    ├── sensibilitat_parametres_model.py     (anàlisi sensibilitat hyperparams)
    ├── comparativa_*.png                    Comparatives globals (estil memòria TFG)
    ├── gradcam/                             Mapes d'atenció Grad-CAM
    └── resnet*_final_*/                     PNGs per model (matrius, mètriques)
```

---

## Instal·lació

Requereix **Python ≥ 3.10**. Es recomana entorn virtual.

```bash
pip install -r requirements.txt
```

Per a la fase de segmentació també cal SAM2 (no està a PyPI):

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

I cal el checkpoint `sam2_hiera_large.pt` dins de `Segmentation/`.

---

## Pipeline complet

```
[1] Descàrrega → [2] Segmentació → [3] Tensor → [4] Entrenament → [5] Avaluació
```

Tots els scripts principals (`train_models.py`, `evaluate_models.py`, `gradcam_attention.py`) s'executen **des de l'arrel del projecte** amb paths relatius a aquesta carpeta.

### 1. Descàrrega dels frames originals

```bash
python Dataset/download_minerals.py
```

Genera `Dataset/original_videos/<Mineral>/{PPL,XPL}/<roca>_rot<N>/frame_XXX.jpg` (72 frames PPL + 72 XPL per regió). Opcions útils:

```bash
python Dataset/download_minerals.py --minerals biotite        # només biotites
python Dataset/download_minerals.py --skip-existing           # reprendre
```

### 2. Segmentació amb SAM2 (interactiu)

```bash
cd Segmentation
python Codi_Segmentacion.py
```

GUI OpenCV. Per cada làmina:

| Tecla | Acció |
|---|---|
| `R` | Seleccionar ROI rectangular sobre el cristall |
| `Enter` | Calcular màscara amb SAM2 |
| `C` | Cancel·lar màscara |
| `P` | Preview rotació 72 frames |
| `S` | Guardar el cristall (genera 72 PNG PPL + 72 XPL) |
| Drag vertical | Rotar la platina manualment |
| `N` | Següent roca |
| `Esc` | Sortir |

Sortida: `Dataset/segmented_minerals/<Mineral>/<roca>_rot<N>_crystal_<id>/{PPL_frames,XPL_frames}/`.

### 3. Conversió a tensors

```bash
cd Segmentation
python TF2Tensor.py
```

Genera `Dataset/segmented_minerals_tensor/<Mineral>/<...>.pt` (vegeu format a `Dataset/METADADES.md`).

### 4. Entrenament dels models

```bash
python train_models.py
```

Menú interactiu. Triant un número [1–7] s'entrena el model corresponent amb 5-fold CV i es guarden els millors checkpoints per F1 a `Models/<results_dir>/fold_<N>/best_model.pth`.

| Model | Bottleneck | Agregació temporal | Backbone |
|---|---|---|---|
| Attention ResNet18 | 1543→512 | self-attention 3 capes | ResNet18 |
| Attention ResNet50 | 6151→512 | self-attention 3 capes | ResNet50 |
| LSTM ResNet18 | 1543→512 | LSTM 1 capa, hidden 512 | ResNet18 |
| LSTM ResNet50 | 6151→512 | LSTM 2 capes, hidden 256 | ResNet50 |
| MeanPooling ResNet18 | 1543→512 | mean pooling temporal | ResNet18 |
| MeanPooling ResNet50 | 6151→512 | mean pooling temporal | ResNet50 |
| StaticImage ResNet18 | 1536→64 | (només frame 0) | ResNet18 |

**Hyperparams comuns**: `BATCH_SIZE=4`, `LR=1e-4`, `MAX_FRAMES=72`, weight decay `1e-4`, balanceig de classes per pèrdua. Els ResNet18 i StaticImage usen `CosineAnnealingLR` i mixed-precision (CUDA).

### 5. Avaluació + visualització

```bash
python evaluate_models.py
```

Menú interactiu:

```
[1] ResNet18 + Attention
[2] ResNet50 + Attention
[3] ResNet18 + LSTM   (guanyador)
[4] ResNet50 + LSTM
[5] ResNet18 + MeanPooling
[6] ResNet50 + MeanPooling
[7] ResNet18 + StaticImage
[8] Avaluar TOTS i generar comparativa
[0] Sortir
```

**Per cada model** (sortida a `Results/<model_dir>/`):
- `matrius_confusio_per_fold.png` — matrius de confusió per fold (absoluta + normalitzada)
- `matriu_confusio_global.png` — suma dels 5 folds
- `metriques_per_fold.png` — Accuracy/F1/Precision/Recall × fold
- `resum_global.png` — barres mitjana ± std

**Pel guanyador (ResNet18 + LSTM)** afegeix:
- `corba_roc.png`
- `distribucio_confianca.png`
- `matriu_confusio_heatmap.png`
- `analisi_errors_visual.png` — graella 4×4 de FP/FN extrems

**Quan s'escull `[8]`**, també a `Results/`:
- `comparativa_models.png` — comparativa global
- `comparativa_resnet18_TFG.png` / `comparativa_resnet50_TFG.png` — estil memòria

### 6. Grad-CAM

```bash
python gradcam_attention.py
```

Genera mapes d'activació Grad-CAM sobre el ResNet50 + Attention a `Results/gradcam/` per a una mostra de Biotita i una de Hornblenda.

---

## Reproduïbilitat

- Llavor global `seed=42` per a `random`, `numpy` i `torch`
- `cudnn.deterministic = True`, `benchmark = False`
- Dispositiu: detecció automàtica `mps` (Apple) > `cuda` (NVIDIA) > `cpu`
- 5-fold estratificat amb `StratifiedKFold(random_state=42)`
- Augmentació només a train (rotació temporal cíclica, flips espacials, brillantor ±10%, soroll Gaussià σ=0.01)

**Nota sobre l'ordre de fitxers**: els 3 ResNet50 originals construïen el dataset amb `os.listdir(...)` directe (ordre del filesystem); els ResNet18 i StaticImage amb `sorted(...)`. `train_models.py` i `evaluate_models.py` mantenen aquesta diferència via el flag `sort_files` per reproduir els splits exactes dels checkpoints originals.

---

## Llicència

Codi sota llicència [Creative Commons Reconeixement-NoComercial-SenseObraDerivada 4.0 Internacional (CC BY-NC-ND 4.0)](http://creativecommons.org/licenses/by-nc-nd/4.0/).

Les imatges originals de microscòpia provenen de l'[Open University Virtual Microscope](https://www.virtualmicroscope.org) i estan subjectes a les seves pròpies condicions d'ús.
