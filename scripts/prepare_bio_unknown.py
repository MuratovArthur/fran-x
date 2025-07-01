#!/usr/bin/env python3
"""
Span → BIO converter  (v5, 2025-06-20)

New in v5
---------
• build_aliases() now returns:
    – every contiguous *suffix* of the entity
      (“Met Office”, “Office”, …)
    – every individual token (≥3 letters)
  ⇒ middle tokens like “Met” propagate correctly.
• Everything else unchanged from v4.
"""

from __future__ import annotations
import argparse, json, os, sys, unicodedata
from typing import List, Tuple, Dict

WINDOW   = 30
ROLE_SET = {"Protagonist", "Antagonist", "Innocent", "Unknown"}

STOP_TOKENS = {
    "the", "a", "an", "of", "for", "in", "on", "at",
    "to", "by", "from", "with", "and", "or", "but",
    "this", "that", "these", "those"
}

# --- Unknown step ---
ALLOWED_NER = {"PERSON", "ORG", "GPE", "NORP", "FAC"}

# ──────────────────────────────────────────────────────────────────────────
# 1) Unicode-aware tokeniser
# ──────────────────────────────────────────────────────────────────────────
try:
    import regex as ure
    TOKEN_RE = ure.compile(r'(?:\p{L}\p{M}*)+|\p{N}+|[^\p{L}\p{N}\s]')
except ModuleNotFoundError:
    import re as ure
    print("ℹ️  Falling back to basic tokeniser; install `regex` for full Unicode support",
          file=sys.stderr)
    TOKEN_RE = ure.compile(r"\w+|[^\w\s]")

def tokenize(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]

# ──────────────────────────────────────────────────────────────────────────
# 2) Gold-annotation helpers
# ──────────────────────────────────────────────────────────────────────────
def load_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return unicodedata.normalize("NFC", f.read().replace("\r\n", "\n"))

def parse_ann(path: str) -> Dict[str, List[Tuple[int, int, str, str]]]:
    res: Dict[str, List[Tuple[int, int, str, str]]] = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            fn, txt, s, e, role, *_ = ln.rstrip("\n").split("\t")
            if role in ROLE_SET:
                res.setdefault(fn, []).append((int(s), int(e), role, txt))
    return res

def align_span(text: str, s0: int, e0: int, span_txt: str):
    if text[s0:e0] == span_txt:
        return s0, e0
    win_s = max(0, s0 - WINDOW)
    win_e = min(len(text), e0 + WINDOW)
    idx   = text[win_s:win_e].find(span_txt)
    if idx == -1:
        return None
    s = win_s + idx
    return s, s + len(span_txt)

# ──────────────────────────────────────────────────────────────────────────
# 3) BIO labelling
# ──────────────────────────────────────────────────────────────────────────
def label_tokens(labels: List[str], tokens, s: int, e: int, role: str):
    first = True
    for i, (_, ts, te) in enumerate(tokens):
        if ts < e and te > s:
            if labels[i] != "O":
                continue
            labels[i] = ("B-" if first else "I-") + role
            first = False

# ──────────────────────────────────────────────────────────────────────────
# 4) Propagation helpers (new alias logic)
# ──────────────────────────────────────────────────────────────────────────
def build_aliases(entity: str) -> set[str]:
    """
    Return alias strings for propagation:
      • every contiguous suffix of the entity
      • every *content* token (not a stop-word, ≥3 letters)
    """
    parts = entity.split()
    suffixes = {" ".join(parts[i:]) for i in range(len(parts))}
    singles  = {tok for tok in parts
                if len(tok) >= 3
                and unicodedata.category(tok[0]).startswith("L")
                and tok.lower() not in STOP_TOKENS}
    return suffixes | singles

def compile_alias_pattern(alias: str):
    escaped = ure.escape(alias).replace(r"\ ", r"\s+")
    return ure.compile(rf'(?<!\w){escaped}(?!\w)')

def propagate_roles(text: str,
                    resolved: List[Tuple[int, int, str, str]],
                    toks, labels) -> None:
    by_ent: Dict[str, List[Tuple[int, int, str]]] = {}
    for s, e, role, txt in resolved:
        by_ent.setdefault(txt, []).append((s, e, role))

    for ent, ann_list in by_ent.items():
        ann_list.sort(key=lambda x: x[0])
        ann_pos = {(s, e) for s, e, _ in ann_list}

        spans: set[Tuple[int, int]] = set()
        for alias in build_aliases(ent):
            pat = compile_alias_pattern(alias)
            spans.update((m.start(), m.end()) for m in pat.finditer(text))

        # longest span first, then earliest
        for s, e in sorted(spans, key=lambda p: (-(p[1]-p[0]), p[0])):
            if (s, e) in ann_pos:
                continue
            role = next((pr for ps, _, pr in reversed(ann_list) if ps < s),
                        ann_list[0][2])
            label_tokens(labels, toks, s, e, role)

# ──────────────────────────────────────────────────────────────────────────
# 5) Unknown-entity discovery   (same as v4)
# ──────────────────────────────────────────────────────────────────────────
def load_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception as e:
        print("⚠️  spaCy not available – Unknown step skipped\n", e, file=sys.stderr)
        return None

NLP = load_spacy()
SKIP_TYPES = {"DATE", "TIME", "PERCENT", "MONEY",
              "QUANTITY", "ORDINAL", "CARDINAL"}

def add_unknown_entities(text: str, toks, labels):
    if NLP is None:
        return
    doc = NLP(text)
    for ent in doc.ents:
        if ent.label_ not in ALLOWED_NER and ent.label_ in SKIP_TYPES:
            continue
        if any(ts < ent.end_char and te > ent.start_char and labels[i] != "O"
               for i, (_, ts, te) in enumerate(toks)):
            continue
        # skip too-long unknown spans
        if len(ent.text.split()) > 4:
            continue
        label_tokens(labels, toks, ent.start_char, ent.end_char, "Unknown")

# ──────────────────────────────────────────────────────────────────────────
# 6) Per-file processing
# ──────────────────────────────────────────────────────────────────────────
def process_file(text: str,
                 spans: List[Tuple[int, int, str, str]],
                 fn: str):
    toks     = tokenize(text)
    labels   = ["O"] * len(toks)
    failures = []
    resolved = []

    # ① gold
    for s0, e0, role, txt in spans:
        aligned = align_span(text, s0, e0, txt)
        if aligned is None:
            failures.append(f"'{txt}' {s0}-{e0}")
            continue
        s, e = aligned
        label_tokens(labels, toks, s, e, role)
        resolved.append((s, e, role, txt))

    # ② propagate
    propagate_roles(text, resolved, toks, labels)

    # ③ unknown side entities
    add_unknown_entities(text, toks, labels)

    # ④ dump
    recs = [{"text": t, "bio_label": labels[i], "start": st, "end": en - 1}
            for i, (t, st, en) in enumerate(toks)]
    return recs, failures

# ──────────────────────────────────────────────────────────────────────────
# 7) CLI
# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="BIO converter with role propagation and Unknown tag")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--annotations-file", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    ann = parse_ann(args.annotations_file)
    os.makedirs(args.out_dir, exist_ok=True)

    total_fail = 0
    for fn, spans in ann.items():
        raw_path = os.path.join(args.raw_dir, fn)
        if not os.path.isfile(raw_path):
            print("❌ missing", fn, file=sys.stderr)
            continue
        text = load_text(raw_path)
        recs, fail = process_file(text, spans, fn)
        total_fail += len(fail)
        if fail:
            print(f"⚠️  {fn}: {len(fail)} unmatched → {', '.join(fail)}",
                  file=sys.stderr)
        out_path = os.path.join(args.out_dir, fn.replace(".txt", ".json"))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)

    if total_fail:
        print(f"\nFinished with {total_fail} unmatched gold spans.", file=sys.stderr)
    else:
        print("\nAll gold spans matched and propagated ✅")

if __name__ == "__main__":
    main()
