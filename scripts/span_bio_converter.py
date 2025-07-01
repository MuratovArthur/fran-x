#!/usr/bin/env python3
"""
Span → BIO converter  (v6.12, 2025-06-21)

Hotfix v6.12
------------
1. **SpaCy initialization** – ensure `NLP` is defined so Unknown tagging runs.
2. **Backward & forward propagation** – unchanged.
3. **Unknown entities** – allow ORG, LOC, skip punctuation-only spans or >4 tokens; no pure-letter check.
4. **Rejection logging** – write all rejected spaCy entities to `unknown_rejections.txt` in `--out-dir` for further analysis.
5. **Statistics** – count total spaCy entities with labels in UNKNOWN_NER and how many of those are accepted.
"""

from __future__ import annotations
import argparse, json, os, sys, unicodedata, re
from typing import List, Tuple, Dict, Set
from collections import Counter
import spacy

# ──────────────────────────────────────────────────────────────────────────
# SpaCy model load
# ──────────────────────────────────────────────────────────────────────────
try:
    NLP = spacy.load("en_core_web_sm")
    print("ℹ️  spaCy loaded successfully", file=sys.stderr)
except Exception:
    print("⚠️  spaCy not available – skipping Unknown tagging", file=sys.stderr)
    NLP = None

WINDOW   = 30
ROLE_SET = {"Protagonist", "Antagonist", "Innocent", "Unknown"}
STOP_TOKENS = {"the","a","an","of","for","in","on","at","to","by","from","with","and","or","but","this","that","these","those"}

# spaCy label sets
PROPAGATE_NER = {"PERSON", "GPE", "NORP"}
UNKNOWN_NER   = PROPAGATE_NER | {"ORG", "LOC"}

# global file handle for rejection logging
REJ_FILE = None

# counters for stats
unknown_candidate_counts = Counter()   # spaCy ents in UNKNOWN_NER
accepted_unknown_counts = Counter()    # spaCy ents actually labeled

# ──────────────────────────────────────────────────────────────────────────
# 1) Unicode-aware tokenizer
# ──────────────────────────────────────────────────────────────────────────
try:
    import regex as ure
    TOKEN_RE = ure.compile(r'(?:\p{L}\p{M}*)+|\p{N}+|[^\p{L}\p{N}\s]')
except ModuleNotFoundError:
    import re as ure
    print("ℹ️  Install `regex` for improved Unicode tokenization", file=sys.stderr)
    TOKEN_RE = ure.compile(r"\w+|[^\w\s]")

def tokenize(text: str) -> List[Tuple[str,int,int]]:
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]

# ──────────────────────────────────────────────────────────────────────────
# 2) File helpers
# ──────────────────────────────────────────────────────────────────────────

def load_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return unicodedata.normalize("NFC", f.read().replace("\r\n","\n"))

def parse_ann(path: str) -> Dict[str,List[Tuple[int,int,str,str]]]:
    res: Dict[str,List[Tuple[int,int,str,str]]] = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            fn,txt,s,e,role,*_ = ln.rstrip("\n").split("\t")
            if role in ROLE_SET:
                res.setdefault(fn,[]).append((int(s),int(e),role,txt))
    return res

def align_span(text: str, s0: int, e0: int, span_txt: str):
    if text[s0:e0] == span_txt:
        return s0,e0
    win_s = max(0,s0-WINDOW); win_e = min(len(text),e0+WINDOW)
    idx = text[win_s:win_e].find(span_txt)
    if idx < 0: return None
    s = win_s + idx
    return s, s+len(span_txt)

# ──────────────────────────────────────────────────────────────────────────
# 3) Alias helpers
# ──────────────────────────────────────────────────────────────────────────

def build_aliases(entity: str) -> Set[str]:
    parts   = entity.split()
    suffixes= {" ".join(parts[i:]) for i in range(len(parts))}
    singles = {tok for tok in parts if len(tok)>=3 and tok.lower() not in STOP_TOKENS}
    return suffixes | singles

try:
    re_lib = ure
except NameError:
    import re as re_lib

def _compile(alias: str):
    esc = re_lib.escape(alias).replace(r"\ ",r"\s+")
    return re_lib.compile(rf"(?<!\w){esc}(?!\w)")

# ──────────────────────────────────────────────────────────────────────────
# 4) Provenance-aware labeling
# ──────────────────────────────────────────────────────────────────────────

def _write(i:int,new_lbl:str,kind:str,labels:List[str],src:List[str]):
    if kind=="gold" or src[i]!="gold":
        labels[i], src[i] = new_lbl, ("gold" if kind=="gold" else "prop")

def label_span(tokens, s:int, e:int, role:str, kind:str,
               labels:List[str], src:List[str]):
    first = True
    for i,(_,ts,te) in enumerate(tokens):
        if ts<e and te> s:
            _write(i,("B-" if first else "I-")+role,kind,labels,src)
            first=False

# ──────────────────────────────────────────────────────────────────────────
# 5) Clustering logic
# ──────────────────────────────────────────────────────────────────────────

def _token_set(s:str)->Set[str]: return set(s.lower().split())

def _should_merge(alias:str,aliases:Set[str])->bool:
    a_toks=_token_set(alias)
    for o in aliases:
        o_toks=_token_set(o)
        if alias==o and len(a_toks)>=2: return True
        if a_toks<=o_toks or o_toks<=a_toks: return True
    return False

def build_clusters(resolved:List[Tuple[int,int,str,str]]):
    alias_map:Dict[str,int]={}
    clusters:List[Dict[str,object]]=[]
    for s,e,role,txt in resolved:
        aliases=build_aliases(txt)|{txt}
        targets={cid for a,cid in alias_map.items() if _should_merge(a,aliases)}
        if not targets:
            cid=len(clusters)
            clusters.append({"aliases":set(aliases),"spans":[(s,e,role,txt)]})
        else:
            cid=min(targets)
            for oid in sorted(targets-{cid},reverse=True):
                clusters[cid]["aliases"].update(clusters[oid]["aliases"])
                clusters[cid]["spans"].extend(clusters[oid]["spans"])
                clusters.pop(oid)
                for a,c in list(alias_map.items()):
                    if c==oid: alias_map[a]=cid
                    elif c>oid: alias_map[a]=c-1
            clusters[cid]["aliases"].update(aliases)
            clusters[cid]["spans"].append((s,e,role,txt))
        for a in aliases: alias_map[a]=cid
    return clusters

def _allows_fuzzy(ent_txt:str)->bool:
    if NLP is None: return True
    ents = NLP(ent_txt).ents
    return bool(ents and ents[0].label_ in PROPAGATE_NER)

# ──────────────────────────────────────────────────────────────────────────
# 6) Main per-file processing
# ──────────────────────────────────────────────────────────────────────────

def process_file(text:str,spans:List[Tuple[int,int,str,str]],fn:str):
    tokens=tokenize(text); labels=["O"]*len(tokens); src=[""]*len(tokens)
    failures:List[str]=[]; resolved:List[Tuple[int,int,str,str]]=[]
    for s0,e0,role,txt in spans:
        aligned=align_span(text,s0,e0,txt)
        if not aligned: failures.append(f"'{txt}' {s0}-{e0}")
        else:
            s,e=aligned
            label_span(tokens,s,e,role,"gold",labels,src)
            resolved.append((s,e,role,txt))

    # propagate within-document mentions
    clusters = build_clusters(resolved)
    prop_count = 0
    for cl in clusters:
        cl["spans"].sort(key=lambda x: x[0])
        full, frole = cl["spans"][0][3], cl["spans"][0][2]
        pats = [ _compile(full) ]
        if _allows_fuzzy(full):
            pats += [ _compile(a) for a in cl["aliases"] if len(a.split())==1 ]

        # backward prop
        fs, fe, _, _ = cl["spans"][0]
        for pat in pats:
            for m in pat.finditer(text):
                ms, me = m.start(), m.end()
                if me > fs or any(ms<se and me>ss for ss,se,_,_ in cl["spans"]): continue
                first_tok = True
                for i,(_,ts,te) in enumerate(tokens):
                    if ts<me and te>ms and src[i]!="gold":
                        _write(i, ("B-" if first_tok else "I-")+frole, "prop", labels, src)
                        first_tok = False

        # forward prop
        for s,e,role,txt in cl["spans"]:
            current = role
            for pat in pats:
                for m in pat.finditer(text, pos=e):
                    ms, me = m.start(), m.end()
                    if any(ms<se and me>ss for ss,se,_,_ in cl["spans"] if ss>=e): continue
                    first_tok = True
                    for i,(_,ts,te) in enumerate(tokens):
                        if ts<me and te>ms and src[i]!="gold":
                            prev = src[i]
                            _write(i, ("B-" if first_tok else "I-")+current, "prop", labels, src)
                            if src[i]=="prop" and prev!="prop":
                                prop_count += 1
                            first_tok = False

    if NLP:
        doc = NLP(text)
        total_ents = 0
        overlapped = 0

        for ent in doc.ents:
            total_ents += 1
            # count every spaCy entity whose label is in UNKNOWN_NER
            if ent.label_ in UNKNOWN_NER:
                unknown_candidate_counts[ent.label_] += 1

            # existing rejection logic
            reason = None
            if ent.label_ not in UNKNOWN_NER:
                reason = f"label_{ent.label_} not in UNKNOWN_NER"
            elif re.fullmatch(r"[^\w\s]+", ent.text):
                reason = f"text '{ent.text}' is punctuation only"
            elif len(ent.text.split()) > 4:
                reason = f"text '{ent.text}' token-count >4"
            elif ent.text.lower() in STOP_TOKENS:
                reason = f"stop_word '{ent.text}'"

            if reason:
                print(f"[UNK-REJ] {fn}\t{ent.text}\t{reason}", file=REJ_FILE)
                continue

            # check overlap with any gold token
            # find any token whose src[i]=="gold" that falls in this ent
            ms, me = ent.start_char, ent.end_char
            if any(src[i]=="gold" and ts < me and te > ms for i,(_,ts,te) in enumerate(tokens)):
                overlapped += 1
                # skip labeling — this is your "already annotated" guard
                continue

            # spaCy ent passed all filters → label as Unknown across ALL overlapping tokens
            first_tok = True
            labeled_any = False
            for i,(tok,ts,te) in enumerate(tokens):
                # any token that overlaps the entity span and is currently unlabeled ("O"),
                # and is not punctuation-only and not a stop word
                if (ts < ent.end_char and te > ent.start_char and labels[i] == "O"
                    and not re.fullmatch(r"[^\w\s]+", tok)
                    and tok.lower() not in STOP_TOKENS):
                    # _write(i, ("B-" if first_tok else "I-") + "Unknown", "prop", labels, src)
                    # labeled_any = True
                    # first_tok = False
                    pass

            if labeled_any:
                accepted_unknown_counts[ent.label_] += 1
                print(f"[UNK-LAB] {fn}\t{ent.text}\t{ent.start_char}-{ent.end_char}", file=sys.stderr)

        print(f"  spaCy ents in UNKNOWN_NER: {total_ents}")
        print(f"  of which overlapped gold spans: {overlapped}")

    recs = [{"text":t, "bio_label":labels[i], "start":st, "end":en-1}
            for i,(t,st,en) in enumerate(tokens)]
    return recs, failures, prop_count


def main():
    ap = argparse.ArgumentParser(description="BIO converter v6.12")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--annotations-file", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    global REJ_FILE
    rej_path = os.path.join(args.out_dir, "unknown_rejections.txt")
    os.makedirs(args.out_dir, exist_ok=True)  # Ensure output directory exists
    REJ_FILE = open(rej_path, 'w', encoding='utf-8')

    ann = parse_ann(args.annotations_file)
    os.makedirs(args.out_dir, exist_ok=True)
    file_props: Dict[str,int] = {}
    total_fail = 0

    for fn, spans in ann.items():
        raw_path = os.path.join(args.raw_dir, fn)
        if not os.path.isfile(raw_path):
            print(f"❌ missing {fn}", file=sys.stderr)
            continue

        text = load_text(raw_path)
        recs, fails, pc = process_file(text, spans, fn)
        total_fail += len(fails)
        file_props[fn] = pc
        if fails:
            print(f"⚠️ {fn}: {len(fails)} unmatched → {', '.join(fails)}", file=sys.stderr)

        out_path = os.path.join(args.out_dir, fn.replace('.txt','.json'))
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)

    REJ_FILE.close()

    # summary
    print("\nTop 5 files by propagation count:", file=sys.stderr)
    for fn, c in sorted(file_props.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {fn}: {c}", file=sys.stderr)
    if total_fail:
        print(f"\nFinished with {total_fail} unmatched gold spans.", file=sys.stderr)
    else:
        print("\nAll gold spans matched and propagated ✅", file=sys.stderr)

    # new stats
    print(f"\nSpaCy entities seen with labels in UNKNOWN_NER:", file=sys.stderr)
    for label, count in unknown_candidate_counts.most_common():
        print(f"  {label}: {count}", file=sys.stderr)

    print(f"\nSpaCy entities actually accepted as Unknown:", file=sys.stderr)
    for label, count in accepted_unknown_counts.most_common():
        print(f"  {label}: {count}", file=sys.stderr)


if __name__ == '__main__':
    main()