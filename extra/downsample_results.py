import argparse
import cv2
import os
from glob import glob

def get_dst_path(src_path: str, src_dir: str, dst_dir: str) -> str:
    """Mirror `src_dir` tree under `dst_dir`."""
    rel_path = os.path.relpath(src_path, src_dir)
    dst_path = os.path.join(dst_dir, rel_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    return dst_path


def downsample_video(src_path: str, dst_path: str) -> None:
    print(f"Processing video: {src_path}")
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        print(f"Failed to open video {src_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")

    out = cv2.VideoWriter(dst_path, fourcc, fps, (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        downsampled = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        out.write(downsampled)

    cap.release()
    out.release()


def downsample_image(src_path: str, dst_path: str) -> None:
    print(f"Processing image: {src_path}")
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to load image {src_path}")
        return

    height, width = img.shape[:2]
    new_size = (width // 2, height // 2)
    downsampled = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst_path, downsampled)


def main() -> None:
    parser = argparse.ArgumentParser(description="Downsample GOPRO result images/videos by 2x.")
    parser.add_argument("--src", required=True, help="Source directory (mirrored tree will be copied).")
    parser.add_argument("--dst", required=True, help="Destination directory.")
    args = parser.parse_args()

    src_dir = os.path.abspath(args.src)
    dst_dir = os.path.abspath(args.dst)
    os.makedirs(dst_dir, exist_ok=True)

    # Downsample videos
    video_paths = glob(os.path.join(src_dir, "**", "*.mp4"), recursive=True)
    for src_path in video_paths:
        dst_path = get_dst_path(src_path, src_dir=src_dir, dst_dir=dst_dir)
        downsample_video(src_path, dst_path)

    # Downsample images
    image_paths = glob(os.path.join(src_dir, "**", "*.png"), recursive=True)
    for src_path in image_paths:
        dst_path = get_dst_path(src_path, src_dir=src_dir, dst_dir=dst_dir)
        downsample_image(src_path, dst_path)

    print("✅ Downsampling complete.")


if __name__ == "__main__":
    main()
