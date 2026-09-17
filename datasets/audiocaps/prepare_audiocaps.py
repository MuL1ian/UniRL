"""Convert official AudioCaps captions to UniRL prompt JSONL files."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import urllib.request
from typing import Dict, Iterable, List

DEFAULT_SOURCE = "https://raw.githubusercontent.com/cdjkim/audiocaps/d004db3ea1b01cf4fd0347dd8d27db90cadc8809"


def _read_rows(source: str, split: str) -> List[Dict[str, str]]:
    filename = f"{split}.csv"
    if os.path.isdir(source):
        with open(os.path.join(source, filename), encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    url = f"{source.rstrip('/')}/dataset/{filename}"
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - user-selectable dataset source
        content = response.read().decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(content)))


def _select_rows(
    rows: List[Dict[str, str]],
    limit: int,
    seed: int,
    *,
    one_caption_per_clip: bool,
) -> List[Dict[str, str]]:
    valid = [row for row in rows if str(row.get("caption") or "").strip()]
    if one_caption_per_clip:
        rows_by_clip: Dict[tuple[str, str], List[Dict[str, str]]] = {}
        for row in valid:
            rows_by_clip.setdefault((row["youtube_id"], row["start_time"]), []).append(row)
        rng = random.Random(seed)
        valid = [rng.choice(rows_by_clip[clip_key]) for clip_key in sorted(rows_by_clip)]
    if limit <= 0 or limit >= len(valid):
        return valid
    indices = list(range(len(valid)))
    random.Random(seed).shuffle(indices)
    return [valid[index] for index in sorted(indices[:limit])]


def _negative_captions(
    rows: List[Dict[str, str]],
    index: int,
    count: int,
    *,
    seed: int,
    split: str,
) -> List[str]:
    target = rows[index]["caption"].strip()
    rng = random.Random(f"{seed}:{split}:{rows[index]['audiocap_id']}")
    negatives: List[str] = []
    attempted_indices = set()
    while len(attempted_indices) < len(rows):
        candidate_index = rng.randrange(len(rows))
        if candidate_index in attempted_indices:
            continue
        attempted_indices.add(candidate_index)
        caption = rows[candidate_index]["caption"].strip()
        if candidate_index != index and caption != target and caption not in negatives:
            negatives.append(caption)
            if len(negatives) == count:
                return negatives
    raise ValueError(f"Could not find {count} distinct negative captions for AudioCaps row {index}.")


def _write_split(
    rows: List[Dict[str, str]],
    out_path: str,
    *,
    source_split: str,
    negative_count: int,
    seed: int,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as output:
        for index, row in enumerate(rows):
            caption = row["caption"].strip()
            metadata = {
                "audio_caption": caption,
                "negative_audio_captions": _negative_captions(
                    rows,
                    index,
                    negative_count,
                    seed=seed,
                    split=source_split,
                ),
                "audiocap_id": str(row["audiocap_id"]),
                "youtube_id": str(row["youtube_id"]),
                "start_time": float(row["start_time"]),
                "source_dataset": "AudioCaps",
                "source_split": source_split,
            }
            record = {
                "prompt": caption,
                "prompt_id": f"audiocaps:{source_split}:{row['audiocap_id']}",
                "metadata": metadata,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} records -> {out_path}")


def _iter_specs(args: argparse.Namespace) -> Iterable[tuple[str, str, int, int, bool]]:
    yield args.train_split, "train.jsonl", args.train_limit, args.seed, False
    yield args.eval_split, "eval.jsonl", args.eval_limit, args.seed + 1, not args.keep_all_eval_captions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="AudioCaps repo URL or local directory")
    parser.add_argument("--out-dir", default="data/audiocaps")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--train-limit", type=int, default=0, help="0 keeps the full source split")
    parser.add_argument("--eval-limit", type=int, default=64, help="0 keeps the full source split")
    parser.add_argument("--negative-count", type=int, default=7)
    parser.add_argument(
        "--keep-all-eval-captions",
        action="store_true",
        help="keep all five validation captions per clip instead of choosing one deterministically",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.negative_count < 1:
        parser.error("--negative-count must be at least 1")

    for source_split, filename, limit, seed, one_caption_per_clip in _iter_specs(args):
        source_rows = _read_rows(args.source, source_split)
        selected_rows = _select_rows(
            source_rows,
            limit,
            seed,
            one_caption_per_clip=one_caption_per_clip,
        )
        if len(selected_rows) <= args.negative_count:
            parser.error(
                f"{source_split} produced {len(selected_rows)} rows, but --negative-count={args.negative_count} "
                "requires at least one more row"
            )
        _write_split(
            selected_rows,
            os.path.join(args.out_dir, filename),
            source_split=source_split,
            negative_count=args.negative_count,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
