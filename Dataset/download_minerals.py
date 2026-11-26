##############################################################################################################

# © 2026, Ariana Ganea
# Aquesta obra està subjecta a una llicència de Creative Commons Reconeixement-NoComercial-SenseObraDerivada
# 4.0 Internacional (CC BY-NC-ND 4.0). Veure : http://creativecommons.org/licenses/by-nc-nd/4.0/

##############################################################################################################

# CODI UNIFICAT PER DESCARREGAR TOTS ELS MINERALS (BIOTITA + HORNBLENDA)
# Per cada regió de cada làmina baixa 72 frames PPL + 72 frames XPL del microscopi virtual
# (https://www.virtualmicroscope.org)
#
# Estructura de sortida:
#   /original_videos/<Mineral>/{PPL,XPL}/<roca>_rot<N>/frame_XXX.jpg
#
# Ús:
#   python download_minerals.py                       # baixa biotita + hornblenda
#   python download_minerals.py --minerals biotite    # només biotites
#   python download_minerals.py --minerals hornblende # només hornblendes
#   python download_minerals.py --skip-existing       # salta fitxers ja descarregats
#   python download_minerals.py --output /original_videos
#
##############################################################################################################

import argparse
import sys
from pathlib import Path
from time import sleep, time
import requests

##############################################################################
# DATASETS (Mineral, FolderName, AssetID, Rotation)
# Llistes canòniques extretes de Segmentation/Codi_Segmentacion.py
##############################################################################

BIOTITE_DATASETS = [
    ("Biotite", "I08", "jrs004", 2),
    ("Biotite", "I09", "jrs006", 1),
    ("Biotite", "I09", "jrs006", 2),
    ("Biotite", "nepheline_syenite_uk", "nephsyenite1", 2),
    ("Biotite", "nepheline_syenite_uk", "nephsyenite1", 3),
    ("Biotite", "green_schist", "p5049", 1),
    ("Biotite", "green_schist", "p5049", 2),
    ("Biotite", "garnet_mica_schist_galway", "p5215", 1),
    ("Biotite", "garnet_mica_schist_galway", "p5215", 3),
    ("Biotite", "garnet_kyanite_gneiss_slishwood", "p5473", 1),
    ("Biotite", "granophyre_mourne", "ig29", 1),
    ("Biotite", "granophyre_mourne", "ig29", 2),
    ("Biotite", "granite_leinster", "ig1", 1),
    ("Biotite", "granite_leinster", "ig1", 2),
    ("Biotite", "siderophyllite_granite_colcerrow33", "12agt33", 1),
    ("Biotite", "siderophyllite_granite_colcerrow33", "12agt33", 2),
    ("Biotite", "siderophyllite_granite_pelyn32", "12agt32", 3),
    ("Biotite", "siderophyllite_granite_luxulyan2", "137201", 1),
    ("Biotite", "siderophyllite_granite_luxulyan2", "137201", 2),
    ("Biotite", "siderophyllite_granite_luxulyan2", "137201", 3),
    ("Biotite", "M02", "p5468", 1),
    ("Biotite", "M02", "p5468", 2),
    ("Biotite", "siderophyllite_granite_luxulyan1", "11agt10", 2),
    ("Biotite", "altered_granite_schist_contact", "cd3668", 1),
    ("Biotite", "garnet_biotite_augen_gneiss", "cd0468", 1),
    ("Biotite", "garnet_biotite_augen_gneiss", "cd0468", 2),
    ("Biotite", "myrmekitic_granulite_charnockite", "cd0311", 1),
    ("Biotite", "M09", "p5563", 1),
    ("Biotite", "M09", "p5563", 2),
    ("Biotite", "granite_I04", "tc_p5471", 1),
    ("Biotite", "quartz_monzonite_jersey", "smg490", 2),
    ("Biotite", "sample_s339_20", "s339_20", 1),
    ("Biotite", "sample_s339_20", "s339_20", 2),
    ("Biotite", "sample_s339_20", "s339_20", 3),
    ("Biotite", "sample_glahm134581", "glahm134581", 1),
    ("Biotite", "sample_glahm134581", "glahm134581", 2),
    ("Biotite", "sample_bgs_59192", "bgs_59192", 2),
    ("Biotite", "sample_11agt16", "11agt16", 1),
    ("Biotite", "shap_granite", "shap_gr", 1),
    ("Biotite", "shap_granite", "shap_gr", 2),
    ("Biotite", "generic_granite", "granite", 1),
    ("Biotite", "generic_granite", "granite", 2),
    ("Biotite", "generic_granite", "granite", 3),
    ("Biotite", "generic_granite", "granite", 4),
]

HORNBLENDE_DATASETS = [
    ("Hornblende", "I08", "jrs004", 1),
    ("Hornblende", "I09", "jrs006", 1),
    ("Hornblende", "I09", "jrs006", 2),
    ("Hornblende", "M01", "p5067", 1),
    ("Hornblende", "M01", "p5067", 2),
    ("Hornblende", "hornblende_porphyrite_blue_mountains", "cd2142", 1),
    ("Hornblende", "hornblende_porphyrite_valparaiso", "cd2142", 1),
    ("Hornblende", "hornblende_porphyrite_valparaiso", "cd2142", 2),
    ("Hornblende", "diorite_magellan_straits", "cd2079", 1),
    ("Hornblende", "hornblende_schist", "cd0660", 1),
    ("Hornblende", "amygdaloidal_monchiquite", "cd0261", 1),
    ("Hornblende", "amygdaloidal_monchiquite", "cd0261", 2),
    ("Hornblende", "amygdaloidal_monchiquite", "cd0261", 3),
    ("Hornblende", "phonolite_cape_verde", "cd0164", 1),
    ("Hornblende", "epidote_amphibolite_mayo", "p5468", 1),
    ("Hornblende", "epidote_amphibolite_mayo", "p5468", 2),
    ("Hornblende", "I16", "p19184", 1),
    ("Hornblende", "I16", "p19184", 2),
    ("Hornblende", "M26", "p19181", 2),
    ("Hornblende", "felsite_isle_rum", "du13876", 1),
    ("Hornblende", "felsite_isle_rum", "du9907", 2),
    ("Hornblende", "granophyre_isle_rum", "du9904", 1),
]


##############################################################################
# CONFIGURACIÓ
##############################################################################

BASE_URL = "https://www.virtualmicroscope.org/sites/default/files/html5Assets"
MODES = ["ppl", "xpl"]
FRAMES_PER_REGION = 72  # 1 frame cada 5° → gir complet 360°
REQUEST_TIMEOUT = 10
SLEEP_BETWEEN_FRAMES = 0.1  # ser amables amb el servidor


##############################################################################
# FUNCIONS
##############################################################################

def build_frame_url(asset_id: str, rotation: int, mode: str, frame_idx: int) -> tuple[str, str]:
    """Construeix la URL d'un frame i el nom local."""
    region = f"r{rotation}"
    folder_name = f"{asset_id}_{region}_{mode}"
    filename = f"{folder_name}{frame_idx:03d}.jpg"
    url = f"{BASE_URL}/{asset_id}/data/{folder_name}/{filename}"
    return url, filename


def download_region(mineral: str, rock_name: str, asset_id: str, rotation: int,
                    output_root: Path, skip_existing: bool) -> dict:
    """Descarrega les 72×2 imatges (PPL + XPL) d'una sola regió.

    Retorna un dict amb el comptatge de fitxers descarregats / saltats / fallits.
    """
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    for mode in MODES:
        save_dir = output_root / mineral / mode.upper() / f"{rock_name}_rot{rotation}"
        save_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, FRAMES_PER_REGION + 1):
            url, _ = build_frame_url(asset_id, rotation, mode, i)
            local_path = save_dir / f"frame_{i:03d}.jpg"

            if skip_existing and local_path.exists() and local_path.stat().st_size > 0:
                stats["skipped"] += 1
                continue

            try:
                r = requests.get(url, timeout=REQUEST_TIMEOUT)
            except requests.RequestException as e:
                print(f"   ✗ frame {i:02d} {mode.upper()}: error xarxa ({e})")
                stats["failed"] += 1
                # No trenquem: el següent frame podria existir
                sleep(SLEEP_BETWEEN_FRAMES)
                continue

            if r.status_code == 200 and r.content:
                local_path.write_bytes(r.content)
                stats["downloaded"] += 1
            elif r.status_code == 404:
                # Fi natural de la sequència
                print(f"   · frame {i:02d} {mode.upper()}: no existeix (final)")
                break
            else:
                print(f"   ✗ frame {i:02d} {mode.upper()}: HTTP {r.status_code}")
                stats["failed"] += 1

            sleep(SLEEP_BETWEEN_FRAMES)

    return stats


def download_dataset(datasets: list, output_root: Path, skip_existing: bool):
    """Descarrega tota una llista de regions."""
    total = len(datasets)
    grand_stats = {"downloaded": 0, "skipped": 0, "failed": 0, "regions_ok": 0}
    start = time()

    for idx, (mineral, rock_name, asset_id, rotation) in enumerate(datasets, start=1):
        region_label = f"{rock_name}_rot{rotation}"
        print(f"\n[{idx:>2}/{total}] {mineral:<10} {region_label:<55} (asset={asset_id})")

        stats = download_region(mineral, rock_name, asset_id, rotation,
                                output_root, skip_existing)

        for k in ("downloaded", "skipped", "failed"):
            grand_stats[k] += stats[k]

        ok = stats["failed"] == 0 and (stats["downloaded"] + stats["skipped"]) > 0
        if ok:
            grand_stats["regions_ok"] += 1
        marker = "✔" if ok else "✗"
        print(f"   {marker} baixats={stats['downloaded']:3d}  "
              f"saltats={stats['skipped']:3d}  fallits={stats['failed']:3d}")

    elapsed = time() - start
    return grand_stats, elapsed


def print_summary(label: str, stats: dict, total_regions: int, elapsed: float):
    print("\n" + "=" * 72)
    print(f" RESUM — {label}")
    print("=" * 72)
    print(f"  Regions totals:    {total_regions}")
    print(f"  Regions completes: {stats['regions_ok']}")
    print(f"  Frames baixats:    {stats['downloaded']}")
    print(f"  Frames saltats:    {stats['skipped']}")
    print(f"  Frames fallits:    {stats['failed']}")
    print(f"  Temps:             {elapsed:.1f} s")
    print("=" * 72)


##############################################################################
# MAIN
##############################################################################

def parse_args():
    p = argparse.ArgumentParser(
        description="Descarrega frames del microscopi virtual per al dataset de minerals (TFG)."
    )
    p.add_argument(
        "--minerals", nargs="+", default=["biotite", "hornblende"],
        choices=["biotite", "hornblende"],
        help="Quins minerals descarregar (per defecte tots dos).",
    )
    # Ruta per defecte: <repo_root>/Dataset/original_videos, robusta a la ubicació del script
    default_output = Path(__file__).resolve().parent.parent / "Dataset" / "original_videos"
    p.add_argument(
        "--output", type=Path, default=default_output,
        help=f"Carpeta de sortida (per defecte: {default_output}).",
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="No tornar a baixar fitxers que ja existeixin (re-execucions).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Sortida:       {output_root}")
    print(f"Minerals:      {', '.join(args.minerals)}")
    print(f"Skip existing: {args.skip_existing}")

    # Pre-crear l'arbre de carpetes principal
    for mineral in ("Biotite", "Hornblende"):
        for mode in ("PPL", "XPL"):
            (output_root / mineral / mode).mkdir(parents=True, exist_ok=True)

    todo = []
    if "biotite" in args.minerals:
        todo.append(("Biotita", BIOTITE_DATASETS))
    if "hornblende" in args.minerals:
        todo.append(("Hornblenda", HORNBLENDE_DATASETS))

    for label, datasets in todo:
        print("\n" + "#" * 72)
        print(f"# DESCARREGANT {label.upper()} — {len(datasets)} regions")
        print("#" * 72)
        stats, elapsed = download_dataset(datasets, output_root, args.skip_existing)
        print_summary(label, stats, len(datasets), elapsed)

    print("\nDescàrrega completa.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterromput per l'usuari.")
        sys.exit(1)
