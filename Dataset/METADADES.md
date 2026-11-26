# Metadades del Dataset 

## Resum

Dataset de seqüències de vídeo de làmines primes mineralògiques rotades 360° sota el microscopi petrogràfic, en doble il·luminació (PPL i XPL). S'usa per entrenar models que distingeixen **biotita** d'**hornblenda** a partir de la seva resposta òptica al gir de la platina.

| Camp | Valor |
|---|---|
| Classes | 2 (Biotite=0, Hornblende=1) |
| Mostres totals | **97** cristalls segmentats |
| Distribució | 61 Biotite · 36 Hornblende |
| Frames per mostra | 72 (gir complet 360°, 1 frame cada 5°) |
| Modes òptics | PPL (llum polaritzada plana) + XPL (polaritzadors creuats) |
| Resolució dels frames | 280 × 280 px (RGB) |
| Format final | Tensors PyTorch `.pt` |
| Font primària | [virtualmicroscope.org](https://www.virtualmicroscope.org) |
| Llicència de codi | CC BY-NC-ND 4.0 (© 2026, Ariana Ganea) |

---

## Estructura de carpetes

```
Dataset/
├── original_videos/               # Frames JPG descarregats del microscopi virtual
│   ├── Biotite/
│   │   ├── PPL/<roca>_rot<N>/frame_001.jpg ... frame_072.jpg
│   │   ├── XPL/<roca>_rot<N>/frame_001.jpg ... frame_072.jpg
│   │   └── SegmentationMask/      # (opcional) màscares pre-calculades
│   └── Hornblende/                # Mateixa estructura
│
├── segmented_minerals/            # Cristalls aïllats amb SAM2 (PNG)
│   ├── Biotite/<roca>_rot<N>_crystal_<id>/
│   │   ├── PPL_frames/frame_000.png ... frame_071.png
│   │   └── XPL_frames/frame_000.png ... frame_071.png
│   └── Hornblende/<...>
│
└── segmented_minerals_tensor/     # Tensors PyTorch (els que consumeixen els models)
    ├── Biotite/
    │   └── <roca>_rot<N>_crystal_<id>.pt
    └── Hornblende/
        └── <roca>_rot<N>_crystal_<id>.pt
```

---

## Format dels fitxers `.pt`

Cada fitxer és un dict de PyTorch generat per `Segmentation/TF2Tensor.py`:

```python
{
    "ppl":   torch.Tensor,   # shape (72, 3, 280, 280), float32, normalitzat
    "xpl":   torch.Tensor,   # shape (72, 3, 280, 280), float32, normalitzat
    "label": int,            # 0 = Biotite, 1 = Hornblende
    "name":  str,            # ex: "I08_rot2_crystal_1"
}
```

**Preprocessament aplicat als frames PNG → tensor:**

| Pas | Detall |
|---|---|
| Resize | 280×280 píxels |
| ToTensor | escala de 0–255 a 0.0–1.0 |
| Normalize | mean = `[0.485, 0.456, 0.406]`, std = `[0.229, 0.224, 0.225]` (ImageNet) |

---

## Convenció de noms

Format: `<rock_name>_rot<rotation>_crystal_<id>.pt`

| Camp | Significat |
|---|---|
| `rock_name` | identificador descriptiu de la mostra (ex: `granite_leinster`, `I08`) |
| `rot<N>` | regió de rotació dins de la mateixa làmina (1, 2, 3...). Una mateixa làmina pot tenir diferents zones d'observació |
| `crystal_<id>` | índex del cristall segmentat dins d'aquella regió (1, 2, 3...) |

Exemple: `I08_rot2_crystal_3.pt` → tercer cristall extret de la segona regió de la làmina `I08`, classificat com a biotita o hornblenda segons la carpeta on resideix.

---

## Distribució per làmina (origen)

Les **97** (97 en XPL + 97 en PPL) mostres provenen d'aproximadament **62 làmines petrogràfiques diferents** (algunes amb múltiples regions/cristalls extrets per làmina). Cada làmina té un `asset_id` corresponent al microscopi virtual de l'Open University.

**Hornblenda — 22 regions de gir** (vegeu `download_minerals.py` per la llista completa):
`I08`, `I09` (×2), `M01` (×2), `hornblende_porphyrite_blue_mountains`, `hornblende_porphyrite_valparaiso` (×2), `diorite_magellan_straits`, `hornblende_schist`, `amygdaloidal_monchiquite` (×3), `phonolite_cape_verde`, `epidote_amphibolite_mayo` (×2), `I16` (×2), `M26`, `felsite_isle_rum` (×2), `granophyre_isle_rum`.

**Biotita — 40 regions de gir** (vegeu `download_minerals.py`):
`I08`, `I09` (×2), `nepheline_syenite_uk` (×2), `green_schist` (×2), `garnet_mica_schist_galway` (×2), `garnet_kyanite_gneiss_slishwood`, `granophyre_mourne` (×2), `granite_leinster` (×2), `siderophyllite_granite_colcerrow33` (×2), `siderophyllite_granite_pelyn32`, `siderophyllite_granite_luxulyan2` (×3), `M02` (×2), `siderophyllite_granite_luxulyan1`, `altered_granite_schist_contact`, `garnet_biotite_augen_gneiss` (×2), `myrmekitic_granulite_charnockite`, `M09` (×2), `granite_I04`, `quartz_monzonite_jersey`, `sample_s339_20` (×3), `sample_glahm134581` (×2), `sample_bgs_59192`, `sample_11agt16`, `shap_granite` (×2), `generic_granite` (×4).

---

## Geometria temporal

- **72 frames** per cada mostra → un frame cada **5°** sobre 360°
- L'angle del frame `i` és `i × (2π / 72)` radians (= `i × 5°`)
- Els models temporals (LSTM, Attention, MeanPooling) consumeixen tota la seqüència; el model de `StaticImage` només usa el frame `0`.

---

## Pipeline de generació

```
[1] Dataset/download_minerals.py
       ↓ baixa 72 JPGs PPL + 72 JPGs XPL per cada mineral

       Dataset/original_videos/<Mineral>/{PPL,XPL}/<roca>_rot<N>/

[2] Segmentation/Codi_Segmentacion.py  (interactiu — usa SAM2)
       ↓ ROI manual + tracking de la màscara mentre gira la platina

       Dataset/segmented_minerals/<Mineral>/<roca>_rot<N>_crystal_<id>/

[3] Segmentation/TF2Tensor.py
       ↓ resize 280×280 + normalització ImageNet + serialització

       Dataset/segmented_minerals_tensor/<Mineral>/<...>.pt
```

---

## Distribució de classes per a l'entrenament

Tots els scripts d'entrenament fan servir **balanceig de pèrdua** (`CrossEntropyLoss(weight=...)`) basat en aquests recomptes:

```python
class_counts = torch.tensor([61, 36], dtype=torch.float32)  # [Biotite, Hornblende]
weights = (1 / class_counts)
weights = weights / weights.sum()
```

També s'aplica `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` per garantir que cada fold mantingui la proporció de classes original.

---

## Referència

Dades petrogràfiques cedides en el marc del TFG (Ariana Ganea, 2026), amb frames originals procedents del microscopi virtual ([virtualmicroscope.org](https://www.virtualmicroscope.org)).
