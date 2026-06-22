"""CLI for deterministic parasite detection."""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from posthoc.parasite_detector.detector import detect_parasites
from posthoc.parasite_detector.visualize import save_debug_outputs


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def iter_images(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(file for file in path.rglob("*") if file.suffix.lower() in IMAGE_SUFFIXES)


def label_from_path(path):
    return path.parent.name if path.parent.name in {"parasitized", "uninfected"} else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input image file or image directory.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--save-debug", action="store_true", help="Save overlays and per-image feature CSVs.")
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_path in iter_images(args.input):
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
        result = detect_parasites(image_rgb, true_label=label_from_path(image_path), return_debug=args.save_debug)
        rows.append(
            {
                "image": str(image_path),
                "true_label": result["true_label"],
                "inferred_label": result["inferred_label"],
                "segmentation_status": result["segmentation_status"],
                "num_cells": len(result["cell_masks"]),
                "num_accepted": len(result["accepted_candidates"]),
                "num_rejected": len(result["rejected_candidates"]),
            }
        )
        if args.save_debug:
            save_debug_outputs(result, output / "debug", image_path.stem)

    pd.DataFrame(rows).to_csv(output / "results.csv", index=False)


if __name__ == "__main__":
    main()
