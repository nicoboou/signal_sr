"""Preprocess NIH-NLM thin smear annotations into fixed-size RBC crop PNGs."""

import argparse
import csv
from pathlib import Path

from PIL import Image


SET_DIRS = {"point": "Point Set", "polygon": "Polygon Set"}
LABEL_DIRS = {"Uninfected": ("uninfected", 0), "Parasitized": ("parasitized", 1)}
METADATA_FIELDS = (
    "crop_path",
    "label",
    "label_name",
    "annotation_set",
    "patient_id",
    "image_name",
    "cell_id",
    "shape",
    "center_x",
    "center_y",
    "source_image",
    "source_gt",
)


def selected_sets(name):
    return tuple(SET_DIRS) if name == "both" else (name,)


def safe_name(text):
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text)


def image_dir(patient_dir):
    for name in ("Img", "img"):
        directory = patient_dir / name
        if directory.is_dir():
            return directory
    raise FileNotFoundError(f"Missing image directory under {patient_dir}")


def parse_annotation(line, gt_path):
    parts = line.strip().split(",")
    if len(parts) < 7:
        raise ValueError(f"Malformed annotation in {gt_path}: {line!r}")

    cell_id, label_name, shape = parts[0], parts[1], parts[3]
    if label_name not in LABEL_DIRS:
        return None

    coords = [float(value) for value in parts[5:]]
    if shape == "Point":
        center_x, center_y = coords[:2]
    elif shape == "Polygon":
        xs, ys = coords[0::2], coords[1::2]
        center_x, center_y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    else:
        raise ValueError(f"Unsupported annotation shape {shape!r} in {gt_path}")

    return cell_id, label_name, shape, center_x, center_y


def crop_box(center_x, center_y, crop_size):
    left = round(center_x) - crop_size // 2
    top = round(center_y) - crop_size // 2
    return left, top, left + crop_size, top + crop_size


def build_crops(root, out, crop_size=224, annotation_sets="both", overwrite=False, limit=None, metadata_csv=None):
    root, out = Path(root), Path(out)
    metadata_csv = Path(metadata_csv) if metadata_csv else out / "metadata.csv"
    rows = []

    for class_name in ("uninfected", "parasitized"):
        (out / class_name).mkdir(parents=True, exist_ok=True)

    for set_key in selected_sets(annotation_sets):
        set_dir = root / SET_DIRS[set_key]
        for patient_dir in sorted(path for path in set_dir.iterdir() if path.is_dir()):
            img_dir = image_dir(patient_dir)
            gt_dir = patient_dir / "GT"
            for gt_path in sorted(gt_dir.glob("*.txt")):
                image_path = img_dir / f"{gt_path.stem}.jpg"
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing image for annotation file: {image_path}")

                with Image.open(image_path) as source:
                    source = source.convert("RGB")
                    for line in gt_path.read_text().splitlines()[1:]:
                        parsed = parse_annotation(line, gt_path)
                        if parsed is None:
                            continue
                        cell_id, raw_label_name, shape, center_x, center_y = parsed
                        label_name, label = LABEL_DIRS[raw_label_name]
                        crop_name = f"{set_key}__{safe_name(patient_dir.name)}__{safe_name(gt_path.stem)}__{safe_name(cell_id)}.png"
                        crop_path = out / label_name / crop_name

                        if overwrite or not crop_path.exists():
                            source.crop(crop_box(center_x, center_y, crop_size)).save(crop_path)

                        rows.append(
                            {
                                "crop_path": str(crop_path),
                                "label": label,
                                "label_name": label_name,
                                "annotation_set": set_key,
                                "patient_id": patient_dir.name,
                                "image_name": gt_path.stem,
                                "cell_id": cell_id,
                                "shape": shape,
                                "center_x": center_x,
                                "center_y": center_y,
                                "source_image": str(image_path),
                                "source_gt": str(gt_path),
                            }
                        )
                        if limit is not None and len(rows) >= limit:
                            return write_metadata(metadata_csv, rows)

    return write_metadata(metadata_csv, rows)


def write_metadata(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/projects/compures/datasets/NLM", help="Raw NLM dataset root.")
    parser.add_argument("--out", default="/projects/compures/datasets/NLM/crops", help="Output crop directory.")
    parser.add_argument("--crop-size", type=int, default=224, help="Square crop size in pixels.")
    parser.add_argument("--sets", choices=("both", "point", "polygon"), default="both", help="Annotation set(s) to preprocess.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing crop PNGs.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of crops to write, useful for smoke tests.")
    parser.add_argument("--metadata-csv", default=None, help="Metadata CSV path. Defaults to OUT/metadata.csv.")
    args = parser.parse_args()

    count = build_crops(args.root, args.out, args.crop_size, args.sets, args.overwrite, args.limit, args.metadata_csv)
    print(f"Wrote metadata for {count} NLM crops to {args.metadata_csv or Path(args.out) / 'metadata.csv'}")


if __name__ == "__main__":
    main()
