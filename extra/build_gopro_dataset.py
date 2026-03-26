import os
import argparse
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def gamma_to_linear(img, gamma=2.2):
    return np.power(img, gamma)

def linear_to_gamma(img, gamma=2.2):
    return np.power(img, 1/gamma)

def process_single_blur(args):
    seq_dir, img_files, start_idx, save_dir_blur, save_dir_sharp, window_size = args
    half_window = window_size // 2
    frame_indices = range(start_idx, start_idx + window_size)
    center_idx = start_idx + half_window

    blur_seq_dir = os.path.join(save_dir_blur, os.path.basename(seq_dir))
    sharp_seq_dir = os.path.join(save_dir_sharp, os.path.basename(seq_dir))
    os.makedirs(blur_seq_dir, exist_ok=True)
    os.makedirs(sharp_seq_dir, exist_ok=True)

    blur_img_path = os.path.join(blur_seq_dir, img_files[center_idx])

    # Skip if already exists
    if os.path.exists(blur_img_path):
        return

    imgs = []
    for i in frame_indices:
        img_path = os.path.join(seq_dir, img_files[i])
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = img.astype(np.float32) / 255.0
        img_linear = gamma_to_linear(img)
        imgs.append(img_linear)

    # Average in linear space
    avg_img_linear = np.mean(imgs, axis=0)
    avg_img_gamma = linear_to_gamma(np.clip(avg_img_linear, 0, 1))

    # Save blurred image
    cv2.imwrite(blur_img_path, (avg_img_gamma * 255).astype(np.uint8))

    # Save sharp images
    for i in frame_indices:
        src_path = os.path.join(seq_dir, img_files[i])
        dst_path = os.path.join(sharp_seq_dir, img_files[i])
        if not os.path.exists(dst_path):
            shutil.copyfile(src_path, dst_path)

def process_sequence(seq_dir, save_dir_blur, save_dir_sharp, window_size=7, workers=None):
    img_files = sorted([f for f in os.listdir(seq_dir) if f.endswith('.png')])
    tasks = []

    num_windows = (len(img_files) - window_size) // window_size + 1

    for w in range(num_windows):
        start_idx = w * window_size
        if start_idx + window_size <= len(img_files):
            tasks.append((seq_dir, img_files, start_idx, save_dir_blur, save_dir_sharp, window_size))

    if tasks:
        with Pool(processes=workers or cpu_count()) as pool:
            list(tqdm(pool.imap_unordered(process_single_blur, tasks), total=len(tasks), desc=os.path.basename(seq_dir)))

def process_split(src_root, save_root, split=None, window_size=7, workers=None):
    split_src_root = os.path.join(src_root, split) if split else src_root
    if split and not os.path.isdir(split_src_root):
        print(f"Skipping split '{split}': source path does not exist -> {split_src_root}")
        return

    split_save_root = os.path.join(save_root, split) if split else save_root
    save_root_blur = os.path.join(split_save_root, "blur")
    save_root_sharp = os.path.join(split_save_root, "sharp")

    seq_dirs = [
        os.path.join(split_src_root, d)
        for d in os.listdir(split_src_root)
        if os.path.isdir(os.path.join(split_src_root, d))
    ]

    for seq_dir in seq_dirs:
        process_sequence(
            seq_dir,
            save_root_blur,
            save_root_sharp,
            window_size=window_size,
            workers=workers,
        )

def main():
    parser = argparse.ArgumentParser(
        description="Build blurred/sharp GOPRO-style dataset from source sequences."
    )
    parser.add_argument(
        "--src-root",
        required=True,
        help="Source root. Can contain split folders (e.g. train/test) or sequence folders directly.",
    )
    parser.add_argument(
        "--save-root",
        required=True,
        help="Output root where blur/sharp folders will be created.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Optional split names under src-root (example: --splits train test).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=7,
        help="Number of consecutive frames to average for blur generation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="Number of worker processes for parallel frame processing.",
    )
    args = parser.parse_args()

    src_root = os.path.abspath(args.src_root)
    save_root = os.path.abspath(args.save_root)
    os.makedirs(save_root, exist_ok=True)

    if args.splits:
        splits = args.splits
    elif all(os.path.isdir(os.path.join(src_root, s)) for s in ["train", "test"]):
        splits = ["train", "test"]
    else:
        splits = [None]

    for split in splits:
        split_name = split if split else "root"
        print(f"Processing {split_name}...")
        process_split(
            src_root=src_root,
            save_root=save_root,
            split=split,
            window_size=args.window_size,
            workers=args.workers,
        )

if __name__ == "__main__":
    main()
