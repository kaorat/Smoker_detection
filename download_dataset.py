"""
download_dataset.py — Pull the augmented smoker dataset from Google Drive.

Mirrors the Colab notebook's source:
    /content/drive/MyDrive/smoker_project/augmented_dataset
Expected final layout:
    dataset/
        Not_Smoking/
        Smoking/

Usage:
    # If you have the Drive FOLDER URL or ID for augmented_dataset
    python download_dataset.py --folder_id <GOOGLE_DRIVE_FOLDER_ID>

    # If you have a zipped FILE URL or ID instead
    python download_dataset.py --zip_id <GOOGLE_DRIVE_FILE_ID>
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

import gdown

parser = argparse.ArgumentParser()
parser.add_argument("--folder_id", help="Google Drive folder ID (augmented_dataset)")
parser.add_argument("--zip_id", help="Google Drive file ID of a zipped dataset")
parser.add_argument("--out_dir", default="./dataset", help="Where to place dataset")
args = parser.parse_args()

OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

if not args.folder_id and not args.zip_id:
    print("Provide either --folder_id (a Drive folder) or --zip_id (a zip file).")
    print("Tip: share the augmented_dataset folder as 'Anyone with the link' first.")
    sys.exit(1)

if args.folder_id:
    print(f"Downloading folder {args.folder_id} -> {OUT} ...")
    gdown.download_folder(
        id=args.folder_id,
        output=str(OUT),
        quiet=False,
        use_cookies=False,
    )
else:
    zip_path = OUT / "_dataset.zip"
    print(f"Downloading zip {args.zip_id} -> {zip_path} ...")
    gdown.download(id=args.zip_id, output=str(zip_path), quiet=False)
    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(OUT)
    zip_path.unlink()

# Sanity check
expected = {"Not_Smoking", "Smoking"}
present = {p.name for p in OUT.iterdir() if p.is_dir()}
if not expected.issubset(present):
    nested = [p for p in OUT.rglob("*") if p.is_dir() and p.name in expected]
    if nested:
        root = nested[0].parent
        print(f"\nClass folders found one level deep at {root}.")
        print("Move them up so structure is dataset/Not_Smoking and dataset/Smoking.")
    else:
        print(f"\nWARNING: expected class folders {expected}, got {present}")
else:
    n_smoke = sum(1 for _ in (OUT / "Smoking").rglob("*") if _.is_file())
    n_not = sum(1 for _ in (OUT / "Not_Smoking").rglob("*") if _.is_file())
    print(f"\nReady: {n_smoke} Smoking | {n_not} Not_Smoking files in {OUT}")
