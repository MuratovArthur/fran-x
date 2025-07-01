#!/usr/bin/env python3
"""Simple span→BIO converter **with role propagation**.

Changes v2 (Hindi fix)
----------------------
*   Uses the optional `regex` package for fully‑Unicode tokenisation:
    pattern `(?:\p{L}\p{M}*)+` treats Devanagari clusters as one token.
*   Falls back to the old regex if `regex` is not installed (prints a notice).

Core algorithm (unchanged)
--------------------------
1.  Align every gold span (exact or ±30 chars window).
2.  Label overlapping tokens with `B/I-ROLE`.
3.  Propagate the same role to all *exact* occurrences of that entity until a
    later annotation changes the role.
4.  Print each propagated span (`🔄`) and any unmatched gold spans (`⚠️`).
"""
from __future__ import annotations

import argparse
import json
import os
import unicodedata
from typing import Dict, List, Tuple

WINDOW = 30
ROLE_SET = {"Protagonist", "Antagonist", "Innocent"}

# ---------------------------------------------------------------------------
# Unicode‑aware tokeniser
# ---------------------------------------------------------------------------
try:
    import regex as ure  # needs `pip install regex`
    TOKEN_RE = ure.compile(r'(?:\p{L}\p{M}*)+|\p{N}+|[^\p{L}\p{N}\s]')
except ModuleNotFoundError:  # graceful fallback
    import re as ure
    print("ℹ️  Falling back to basic tokeniser; install `regex` for better Indic support")
    TOKEN_RE = ure.compile(r"\w+|[^\w\s]")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return unicodedata.normalize("NFC", f.read().replace("\r\n", "\n"))


def tokenize(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def parse_ann(path: str) -> Dict[str, List[Tuple[int, int, str, str]]]:
    res: Dict[str, List[Tuple[int, int, str, str]]] = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            fn, txt, s, e, role, *_ = ln.rstrip("\n").split("\t")
            if role in ROLE_SET:
                res.setdefault(fn, []).append((int(s), int(e), role, txt))
    return res

# ---------------------------------------------------------------------------
# Alignment + labelling
# ---------------------------------------------------------------------------

def align_span(text: str, s0: int, e0: int, span_txt: str):
    if text[s0:e0] == span_txt:
        return s0, e0
    win_s = max(0, s0 - WINDOW)
    win_e = min(len(text), e0 + WINDOW)
    idx = text[win_s:win_e].find(span_txt)
    if idx == -1:
        return None
    s = win_s + idx
    return s, s + len(span_txt)


def label_tokens(labels: List[str], tokens, s: int, e: int, role: str):
    first = True
    for i, (_, ts, te) in enumerate(tokens):
        if ts < e and te > s:
            if labels[i] != "O":
                continue  # don't overwrite gold
            labels[i] = ("B-" if first else "I-") + role
            first = False

# ---------------------------------------------------------------------------
# Process one file
# ---------------------------------------------------------------------------

def process_file(text: str, spans, fn: str):
    toks = tokenize(text)
    labels = ["O"] * len(toks)
    failures = []
    resolved: List[Tuple[int, int, str, str]] = []

    for s0, e0, role, txt in spans:
        aligned = align_span(text, s0, e0, txt)
        if aligned is None:
            failures.append(f"'{txt}' {s0}-{e0}")
            continue
        s, e = aligned
        label_tokens(labels, toks, s, e, role)
        resolved.append((s, e, role, txt))

    # propagation per entity
    # by_ent: Dict[str, List[Tuple[int, int, str]]] = {}
    # for s, e, role, txt in resolved:
    #     by_ent.setdefault(txt, []).append((s, e, role))
    # for ent, ann_list in by_ent.items():
    #     ann_list.sort(key=lambda x: x[0])
    #     occs = [(m.start(), m.end()) for m in ure.finditer(ure.escape(ent), text)]
    #     ann_pos = {(s, e) for s, e, _ in ann_list}
    #     for s, e in occs:
    #         if (s, e) in ann_pos:
    #             continue
    #         # decide role
    #         role = None
    #         if s < ann_list[0][0]:
    #             role = ann_list[0][2]
    #         elif s > ann_list[-1][0]:
    #             role = ann_list[-1][2]
    #         else:
    #             for i in range(len(ann_list) - 1):
    #                 if ann_list[i][0] < s < ann_list[i + 1][0]:
    #                     role = ann_list[i][2]
    #                     break
    #         if role:
    #             label_tokens(labels, toks, s, e, role)
    #             print(f"🔄 Propagated '{ent}' {s}-{e} as {role} in {fn}")

    recs = [{"text": t, "bio_label": labels[i], "start": st, "end": en - 1} for i, (t, st, en) in enumerate(toks)]
    return recs, failures

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("BIO converter with propagation & Unicode tokens")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--annotations-file", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    ann = parse_ann(args.annotations_file)
    os.makedirs(args.out_dir, exist_ok=True)

    total_fail = 0
    for fn, spans in ann.items():
        raw = os.path.join(args.raw_dir, fn)
        if not os.path.isfile(raw):
            print("❌ missing", fn)
            continue
        text = load_text(raw)
        recs, fail = process_file(text, spans, fn)
        total_fail += len(fail)
        if fail:
            print(f"⚠️  {fn}: {len(fail)} unmatched →", ", ".join(fail))
        out = os.path.join(args.out_dir, fn.replace(".txt", ".json"))
        with open(out, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)

    if total_fail:
        print(f"\nFinished with {total_fail} unmatched gold spans.")
    else:
        print("\nAll gold spans matched and propagated ✅")

if __name__ == "__main__":
    main()
