"""
Truncate sequences in a sequential dataset to the first N images and their labels.

Usage:
    uv run python scripts/truncate_sequences.py \
        --input data/processed/sequential_train_val_cut20_2.0.0 \
        --output data/processed/sequential_train_val_cut20_trunc20_2.0.0 \
        --max-images 20
"""

import argparse
import shutil
from pathlib import Path


def truncate_sequences(input_dir: Path, output_dir: Path, max_images: int) -> None:
    splits = [d for d in input_dir.iterdir() if d.is_dir()]

    for split_dir in sorted(splits):
        categories = [d for d in split_dir.iterdir() if d.is_dir()]
        for category_dir in sorted(categories):
            sequences = [d for d in category_dir.iterdir() if d.is_dir()]
            for seq_dir in sorted(sequences):
                images_dir = seq_dir / "images"
                labels_dir = seq_dir / "labels"

                if not images_dir.exists():
                    print(f"  [skip] no images/ in {seq_dir.relative_to(input_dir)}")
                    continue

                images = sorted(images_dir.glob("*.jpg"))
                selected = images[:max_images]

                out_seq = output_dir / seq_dir.relative_to(input_dir)
                out_images = out_seq / "images"
                out_labels = out_seq / "labels"
                out_images.mkdir(parents=True, exist_ok=True)
                if labels_dir.exists():
                    out_labels.mkdir(parents=True, exist_ok=True)

                for img_path in selected:
                    shutil.copy2(img_path, out_images / img_path.name)
                    if labels_dir.exists():
                        label_path = labels_dir / img_path.with_suffix(".txt").name
                        if label_path.exists():
                            shutil.copy2(label_path, out_labels / label_path.name)

                print(
                    f"  {seq_dir.relative_to(input_dir)}: "
                    f"{len(selected)}/{len(images)} images"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Truncate sequences to first N images.")
    parser.add_argument("--input", type=Path, required=True, help="Input dataset directory")
    parser.add_argument("--output", type=Path, required=True, help="Output dataset directory")
    parser.add_argument("--max-images", type=int, default=20, help="Max images per sequence")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input}")

    print(f"Input:      {args.input}")
    print(f"Output:     {args.output}")
    print(f"Max images: {args.max_images}")
    print()

    truncate_sequences(args.input, args.output, args.max_images)
    print("\nDone.")


if __name__ == "__main__":
    main()
