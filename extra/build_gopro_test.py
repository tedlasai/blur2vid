#!/usr/bin/env python3
"""
Build FullSizeExtra/GTFullSize from GOPRO_7.

Expected input:
  GOPRO_7/<split>/blur/<sequence>/*.png        (center indices, e.g. 000004.png)
  GOPRO_7/<split>/sharp/<sequence>/*.png       (all sharp frames, e.g. 000001.png)

Expected output (written under `--output_root`):
  <sequence>/<center>/frame_00000.png ... frame_00006.png
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gopro7_path",
        type=str,
        required=True,
        help="Path to GOPRO_7 dataset root.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="GTFullSize root folder to write into (e.g. .../FullSizeExtra/GTFullSize).",
    )
    parser.add_argument("--half_window", type=int, default=3, help="Half window size (±N).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing frames.")
    parser.add_argument(
        "--require_all_sharp_exist",
        action="store_true",
        help="If any sharp frame in the window is missing, skip the whole window.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print actions without copying.")
    parser.add_argument(
        "--limit_windows",
        type=int,
        default=0,
        help="Optional cap on number of windows processed (0 = no limit).",
    )
    return parser.parse_args()


def copy_window(
    *,
    sharp_seq_dir: Path,
    out_window_dir: Path,
    center_idx: int,
    half_window: int,
    overwrite: bool,
    dry_run: bool,
    require_all_sharp_exist: bool,
) -> bool:
    """
    Returns True if the window is written (or already present and not overwritten),
    returns False if the window is skipped due to missing sharp frames.
    """
    frame_offsets = list(range(-half_window, half_window + 1))  # e.g. [-3..+3]

    expected_dsts = [
        out_window_dir / f"frame_{i:05d}.png" for i in range(len(frame_offsets))
    ]
    if not overwrite and all(p.exists() for p in expected_dsts):
        return True

    # Ensure output directory exists before checking/writing.
    if not dry_run:
        out_window_dir.mkdir(parents=True, exist_ok=True)

    # Verify availability first if requested.
    if require_all_sharp_exist:
        for i, offset in enumerate(frame_offsets):
            sharp_idx = center_idx + offset
            sharp_src = sharp_seq_dir / f"{sharp_idx:06d}.png"
            if not sharp_src.exists():
                return False

    copied_any = False
    for i, offset in enumerate(frame_offsets):
        sharp_idx = center_idx + offset
        sharp_src = sharp_seq_dir / f"{sharp_idx:06d}.png"
        dst = out_window_dir / f"frame_{i:05d}.png"

        if not sharp_src.exists():
            if require_all_sharp_exist:
                return False
            # Without require_all_sharp_exist, just skip missing frames.
            continue

        if (not overwrite) and dst.exists():
            continue

        if dry_run:
            print(f"[dry-run] copy {sharp_src} -> {dst}")
            copied_any = True
            continue

        shutil.copy2(sharp_src, dst)
        copied_any = True

    return copied_any


def main() -> None:
    args = parse_args()

    gopro7_path = Path(args.gopro7_path)
    blur_root = gopro7_path / args.split / "blur"
    sharp_root = gopro7_path / args.split / "sharp"
    out_root = Path(args.output_root)

    if not blur_root.exists():
        raise FileNotFoundError(f"Missing blur_root: {blur_root}")
    if not sharp_root.exists():
        raise FileNotFoundError(f"Missing sharp_root: {sharp_root}")

    windows_processed = 0
    windows_written = 0
    windows_skipped_missing = 0

    for seq_dir in sorted([p for p in blur_root.iterdir() if p.is_dir()]):
        seq_name = seq_dir.name
        sharp_seq_dir = sharp_root / seq_name
        if not sharp_seq_dir.exists():
            print(f"⚠️  Missing sharp sequence: {sharp_seq_dir} (skip {seq_name})")
            continue

        out_seq_root = out_root / seq_name
        if not args.dry_run:
            out_seq_root.mkdir(parents=True, exist_ok=True)

        blur_files = sorted(seq_dir.glob("*.png"))
        for blur_path in blur_files:
            if args.limit_windows and windows_processed >= args.limit_windows:
                print(f"Reached limit_windows={args.limit_windows}; stopping.")
                print(
                    f"Summary: processed={windows_processed}, written={windows_written}, skipped_missing={windows_skipped_missing}"
                )
                return

            center_idx = int(blur_path.stem)
            out_window_dir = out_seq_root / f"{center_idx:06d}"

            windows_processed += 1
            wrote = copy_window(
                sharp_seq_dir=sharp_seq_dir,
                out_window_dir=out_window_dir,
                center_idx=center_idx,
                half_window=args.half_window,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                require_all_sharp_exist=args.require_all_sharp_exist,
            )
            if wrote:
                windows_written += 1
            else:
                windows_skipped_missing += 1

    print(
        f"Summary: processed={windows_processed}, written={windows_written}, skipped_missing={windows_skipped_missing}"
    )


if __name__ == "__main__":
    main()
