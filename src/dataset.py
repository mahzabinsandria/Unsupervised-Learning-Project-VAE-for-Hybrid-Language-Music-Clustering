from __future__ import annotations
import argparse
import os
import shutil
from pathlib import Path
import urllib.request

import pandas as pd
from datasets import load_dataset, Audio


ROOT = Path(__file__).resolve().parents[1]  # project/
DATA_AUDIO = ROOT / "data" / "audio"
DATA_LYRICS = ROOT / "data" / "lyrics"


BANGLA_CSV_URL = "https://raw.githubusercontent.com/Hamza029/Bangla-Songs-Dataset/main/Bangla-Songs-Dataset.csv"
ENGLISH_HF_DATASET = "brunokreiner/genius-lyrics"
JAMENDO_HF_DATASET = "jamendolyrics/jamendolyrics"


def _download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r:
        out_path.write_bytes(r.read())


def _find_lyrics_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "lyric" in c.lower():
            return c
    raise ValueError(f"Could not find lyrics column. Columns: {list(df.columns)}")


def prepare_easy(n_en: int = 2000, n_bn: int = 1000, seed: int = 42) -> Path:
    """
    Builds: data/lyrics/tracks_easy.csv
    Columns: id, lyrics, language (en/bn)
    """
    DATA_LYRICS.mkdir(parents=True, exist_ok=True)

    # Bangla lyrics (GitHub CSV)
    bn_raw = DATA_LYRICS / "_bangla_raw.csv"
    _download_file(BANGLA_CSV_URL, bn_raw)
    bn = pd.read_csv(bn_raw)
    bn_lyrics_col = _find_lyrics_column(bn)

    bn_df = pd.DataFrame({
        "id": [f"bn_{i}" for i in range(len(bn))],
        "lyrics": bn[bn_lyrics_col].fillna("").astype(str),
        "language": "bn",
    }).sample(n=min(n_bn, len(bn)), random_state=seed)

    # English lyrics (HuggingFace)
    en_ds = load_dataset(ENGLISH_HF_DATASET, split="train")
    en = en_ds.to_pandas()
    if "is_english" in en.columns:
        en = en[en["is_english"] == True].copy()
    if "lyrics" not in en.columns:
        raise ValueError(f"{ENGLISH_HF_DATASET} must contain a 'lyrics' column.")

    en = en.sample(n=min(n_en, len(en)), random_state=seed)
    en_df = pd.DataFrame({
        "id": [f"en_{i}" for i in range(len(en))],
        "lyrics": en["lyrics"].fillna("").astype(str),
        "language": "en",
    })

    out = pd.concat([bn_df, en_df], ignore_index=True)
    out_path = DATA_LYRICS / "tracks_easy.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_av(jamendo_split: str = "test", max_items: int = 300, seed: int = 42) -> Path:
    """
    Builds: data/lyrics/tracks_av.csv
    Copies audio into: data/audio/jamendolyrics/
    Columns: id, audio_path, lyrics, language, genre
    """
    DATA_AUDIO.mkdir(parents=True, exist_ok=True)
    DATA_LYRICS.mkdir(parents=True, exist_ok=True)

    audio_out_dir = DATA_AUDIO / "jamendolyrics"
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(JAMENDO_HF_DATASET, split=jamendo_split)
    # keep audio as paths, not decoded arrays
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.shuffle(seed=seed).select(range(min(max_items, len(ds))))

    rows = []
    for i, item in enumerate(ds):
        src_audio_path = Path(item["audio"]["path"])
        ext = src_audio_path.suffix if src_audio_path.suffix else ".mp3"
        dst = audio_out_dir / f"jam_{i:04d}{ext}"

        if not dst.exists():
            shutil.copy(src_audio_path, dst)

        rows.append({
            "id": f"jam_{i}",
            "audio_path": str(dst.as_posix()),
            "lyrics": str(item.get("text", "")),
            "language": str(item.get("language", "unknown")),
            "genre": str(item.get("genre", "unknown")),
        })

    out = pd.DataFrame(rows)
    out_path = DATA_LYRICS / "tracks_av.csv"
    out.to_csv(out_path, index=False)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="Download/build datasets into data/lyrics and data/audio")
    ap.add_argument("--n_en", type=int, default=2000)
    ap.add_argument("--n_bn", type=int, default=1000)
    ap.add_argument("--jamendo_split", type=str, default="test")
    ap.add_argument("--jamendo_max", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.prepare:
        print("Nothing to do. Run with --prepare")
        return

    easy_path = prepare_easy(n_en=args.n_en, n_bn=args.n_bn, seed=args.seed)
    av_path = prepare_av(jamendo_split=args.jamendo_split, max_items=args.jamendo_max, seed=args.seed)

    print("Saved:", easy_path)
    print("Saved:", av_path)
    print("Audio copied into:", (DATA_AUDIO / "jamendolyrics"))


if __name__ == "__main__":
    main()
