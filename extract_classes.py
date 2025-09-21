#!/usr/bin/env python3
# extract_classes.py
import os, re, shutil, argparse, sys, csv
from pathlib import Path
from collections import defaultdict

# Final target classes (uppercase folder names)
TARGET = {"ANG", "DIS", "FEA", "HAP", "NEU", "SAD"}
UNMAPPED = "UNMAPPED"

# ---------------- Parsers per dataset ----------------


def parse_cremad(fname: str):
    """
    CREMA-D: '1001_DFA_ANG_XX.wav' -> third token is emotion code
    Allowed codes: ANG, DIS, FEA, HAP, NEU, SAD
    """
    stem = Path(fname).stem
    toks = stem.split("_")
    if len(toks) < 3:
        return None
    code = toks[2].upper()
    return code if code in TARGET else None


def parse_ravdess(fname: str):
    """
    RAVDESS: '03-01-05-01-01-01-01.wav'
    third field is emotion code:
      01 NEU, 03 HAP, 04 SAD, 05 ANG, 06 FEA, 07 DIS
      (02 calm, 08 surprise are skipped)
    """
    parts = Path(fname).stem.split("-")
    if len(parts) < 3:
        return None
    emo = parts[2]
    mapping = {
        "01": "NEU",
        "03": "HAP",
        "04": "SAD",
        "05": "ANG",
        "06": "FEA",
        "07": "DIS",
    }
    return mapping.get(emo)


def parse_tess(fname: str):
    """
    TESS: filename contains emotion word (e.g., 'OAF_happy.wav').
    Surprise/pleasant_surprise are skipped.
    """
    name = Path(fname).stem.lower()
    if "pleasant_surprise" in name or "surprise" in name:
        return None
    word_map = {
        "angry": "ANG",
        "disgust": "DIS",
        "fear": "FEA",
        "happy": "HAP",
        "neutral": "NEU",
        "sad": "SAD",
    }
    for w, code in word_map.items():
        if re.search(rf"(?:^|[_-]){w}(?:$|[_-])", name):
            return code
    return None


def parse_savee(fname: str):
    """
    SAVEE prefixes: a=ANG, d=DIS, f=FEA, h=HAP, n=NEU, sa=SAD, su=surprise (skip)
    Examples: 'DC_a01.wav', 'JE_sa12.wav', 'KL_su07.wav'
    """
    stem = Path(fname).stem.lower()
    m = re.search(r"[a-z]{2}_?([a-z]+)\d+", stem)
    if not m:
        return None
    tok = m.group(1)
    if tok.startswith("sa"):
        return "SAD"
    if tok.startswith("su"):
        return None  # skip surprise
    mapping = {"a": "ANG", "d": "DIS", "f": "FEA", "h": "HAP", "n": "NEU"}
    return mapping.get(tok[0])


# Router based on folder hint
def detect_and_parse(path: Path):
    p = "/".join(part.lower() for part in path.parts)
    fname = path.name
    if "crema" in p:
        return parse_cremad(fname)
    if "ravdess" in p:
        return parse_ravdess(fname)
    if "tess" in p:
        return parse_tess(fname)
    if "savee" in p:
        return parse_savee(fname)
    # fallback: try all
    for f in (parse_cremad, parse_ravdess, parse_tess, parse_savee):
        code = f(fname)
        if code:
            return code
    return None


def iter_wavs(root: Path):
    for p in root.rglob("*.wav"):
        if p.is_file():
            yield p


def recount_from_disk(out_root: Path):
    """Ground-truth recount by scanning the output tree."""
    counts = defaultdict(int)
    if not out_root.exists():
        return counts
    for p in out_root.rglob("*.wav"):
        counts[p.parent.name.upper()] += 1
    for k in TARGET | {UNMAPPED}:
        counts[k] += 0
    return counts


def write_counts_csv(out_root: Path, counts: dict, fname="counts.csv"):
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "count"])
        for k in sorted(counts.keys()):
            w.writerow([k, counts[k]])


def main():
    ap = argparse.ArgumentParser(
        description="Extract .wav files into per-class folders (ANG, DIS, FEA, HAP, NEU, SAD) plus UNMAPPED."
    )
    ap.add_argument(
        "--input_root",
        default="data",
        help="Root containing Crema, Ravdess, Savee, Tess",
    )
    ap.add_argument(
        "--output_root", default="extracted", help="Where to put class folders"
    )
    ap.add_argument("--link", action="store_true", help="Symlink instead of copy")
    ap.add_argument("--dry", action="store_true", help="Dry run: parse & count only")
    ap.add_argument("--no_csv", action="store_true", help="Do not write counts.csv")
    args = ap.parse_args()

    in_root = Path(args.input_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    if not in_root.exists():
        print(f"[ERR] Input root not found: {in_root}", file=sys.stderr)
        sys.exit(1)

    # Prepare output dirs
    if not args.dry:
        out_root.mkdir(parents=True, exist_ok=True)
        for cls in sorted(TARGET | {UNMAPPED}):
            (out_root / cls).mkdir(parents=True, exist_ok=True)

    live_counts = defaultdict(int)
    reasons = defaultdict(int)

    for wav in iter_wavs(in_root):
        cls = detect_and_parse(wav)
        if cls is None:
            dest = out_root / UNMAPPED / wav.name
            if not args.dry:
                if args.link:
                    try:
                        if dest.exists():
                            dest.unlink()
                        os.symlink(wav, dest)
                    except OSError:
                        shutil.copy2(wav, dest)
                else:
                    shutil.copy2(wav, dest)
            live_counts[UNMAPPED] += 1
            reasons["unmapped_or_excluded"] += 1
            continue

        dest = out_root / cls / wav.name
        if args.dry:
            live_counts[cls] += 1
            continue
        if args.link:
            try:
                if dest.exists():
                    dest.unlink()
                os.symlink(wav, dest)
            except OSError:
                shutil.copy2(wav, dest)
        else:
            shutil.copy2(wav, dest)
        live_counts[cls] += 1

    print("\nCounts during processing:")
    for k in sorted(TARGET | {UNMAPPED}):
        print(f"  {k:9s}: {live_counts[k]:5d}")
    print(f"\nTotal processed: {sum(live_counts.values())}")
    if reasons:
        print("Reasons (for UNMAPPED):")
        for r, n in reasons.items():
            print(f"  {r}: {n}")

    if not args.dry:
        final_counts = recount_from_disk(out_root)
        print("\nPost-extraction recount (from disk):")
        for k in sorted(final_counts.keys()):
            print(f"  {k:9s}: {final_counts[k]:5d}")
        print(f"\nTotal on disk: {sum(final_counts.values())}")
        if not args.no_csv:
            write_counts_csv(out_root, final_counts)
            print(f"\nWrote counts CSV to: {out_root / 'counts.csv'}")


if __name__ == "__main__":
    main()
