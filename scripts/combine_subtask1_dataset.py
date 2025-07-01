#!/usr/bin/env python3
import os
import shutil
import argparse

def main(input_dir, output_dir, languages):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare output subdirectories
    raw_out = os.path.join(output_dir, 'raw-documents')
    bio_out = os.path.join(output_dir, 'bio')
    os.makedirs(raw_out, exist_ok=True)
    os.makedirs(bio_out, exist_ok=True)

    # Combined annotation file path
    ann_out_file = os.path.join(output_dir, 'subtask-1-annotations.txt')

    # Open combined annotation file for writing
    with open(ann_out_file, 'w', encoding='utf-8') as ann_out:
        for lang in languages:
            lang_dir = os.path.join(input_dir, lang)

            # Combine annotations
            ann_file = os.path.join(lang_dir, 'subtask-1-annotations.txt')
            if os.path.isfile(ann_file):
                with open(ann_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        ann_out.write(line)
            else:
                print(f"⚠️ Annotation file not found for {lang}: {ann_file}")

            # Combine raw documents
            raw_dir = os.path.join(lang_dir, 'raw-documents')
            if os.path.isdir(raw_dir):
                for fname in os.listdir(raw_dir):
                    if fname.endswith('.txt'):
                        src = os.path.join(raw_dir, fname)
                        dst = os.path.join(raw_out, fname)
                        shutil.copy2(src, dst)
            else:
                print(f"⚠️ Raw documents dir not found for {lang}: {raw_dir}")

            # Combine BIO outputs
            bio_dir = os.path.join(lang_dir, 'bio')
            if os.path.isdir(bio_dir):
                for fname in os.listdir(bio_dir):
                    if fname.endswith('.json'):
                        src = os.path.join(bio_dir, fname)
                        dst = os.path.join(bio_out, fname)
                        shutil.copy2(src, dst)
            else:
                print(f"⚠️ BIO dir not found for {lang}: {bio_dir}")

    print(f"✅ Combined dataset created at {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine subtask-1 data across multiple languages"
    )
    parser.add_argument(
        "--input-dir", default="dataset_full/train",
        help="Root directory containing language subfolders"
    )
    parser.add_argument(
        "--output-dir", default="dataset_combined_original",
        help="Directory to create combined dataset"
    )
    parser.add_argument(
        "--languages", nargs="+", default=["BG", "EN", "HI", "PT", "RU"],
        help="List of language codes to combine"
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.languages) 