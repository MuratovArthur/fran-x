#!/usr/bin/env python3
"""
Compute overall Unknown vs. role-token fraction for BIO JSON files.

Usage:
    python compute_unknown_fraction_overall.py --input-dir dataset_last/EN/bio

Scans all JSON files in the directory and totals:
  • role_cnt    = number of tokens labeled Protagonist/Antagonist/Innocent
  • unknown_cnt = number of tokens labeled Unknown (B-Unknown or I-Unknown)
  • ratio       = unknown_cnt / role_cnt

Prints the aggregate counts and ratio.
"""
import os
import json
import argparse
from glob import glob

ROLE_PREFIXES = (
    "B-Protagonist", "I-Protagonist",
    "B-Antagonist", "I-Antagonist",
    "B-Innocent",    "I-Innocent"
)
UNKNOWN_PREFIXES = ("B-Unknown", "I-Unknown")


def analyze_file(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    role_cnt = 0
    unknown_cnt = 0
    for tok in data:
        lbl = tok.get('bio_label', '')
        if any(lbl.startswith(pref) for pref in ROLE_PREFIXES):
            role_cnt += 1
        elif any(lbl.startswith(pref) for pref in UNKNOWN_PREFIXES):
            unknown_cnt += 1

    return role_cnt, unknown_cnt


def main():
    parser = argparse.ArgumentParser(
        description="Compute overall Unknown/role fraction for BIO JSON files"
    )
    parser.add_argument(
        '--input-dir', '-i', default='dataset_last/EN/bio',
        help='Directory with BIO JSON files'
    )
    args = parser.parse_args()

    files = glob(os.path.join(args.input_dir, '*.json'))

    total_roles = 0
    total_unknowns = 0
    for path in files:
        rc, uc = analyze_file(path)
        total_roles += rc
        total_unknowns += uc

    ratio = (total_unknowns / total_roles) if total_roles > 0 else float('inf')

    print(f"Processed {len(files)} files")
    print(f"Total role tokens   : {total_roles}")
    print(f"Total Unknown tokens: {total_unknowns}")
    print(f"Unknown/role ratio  : {ratio:.4f}")

if __name__ == '__main__':
    main()
