from huggingface_hub import snapshot_download
import os
import sys
# Make sure HF_TOKEN is set in your env beforehand:
# export HF_TOKEN=your_hf_token
#get first command line argument


mode = sys.argv[1] if len(sys.argv) > 1 else "outsidephotos"


REPO_ID = "tedlasai/blur2vid"
REPO_TYPE = "model"

# These are the subfolders you previously used as path_in_repo
if mode == "outsidephotos":
    checkpoints = [
        "cogvideox-outsidephotos",
    ]
elif mode == "gopro":
    checkpoints = [
        "cogvideox-gopro-test",
        "cogvideox-gopro-2x-test",
    ]
elif mode == "baist":
    checkpoints = [
        "cogvideox-baist-test",
    ]
elif mode == "full":
    checkpoints = [
        "cogvideox-baist-test",
        "cogvideox-gopro-test",
        "cogvideox-gopro-2x-test",
        "cogvideox-full-test",
        "cogvideox-outsidephotos",
    ]

# This is the root local directory where you want everything saved
#get path of this file
LOCAL_TRAINING_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training")
os.makedirs(LOCAL_TRAINING_ROOT, exist_ok=True)

# Download only those folders from the repo and place them under LOCAL_TRAINING_ROOT
snapshot_download(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    local_dir=LOCAL_TRAINING_ROOT,
    local_dir_use_symlinks=False,
    allow_patterns=[f"{name}/*" for name in checkpoints],
    token=os.getenv("HF_TOKEN"),
)

print(f"Done! Checkpoints downloaded under: {LOCAL_TRAINING_ROOT}")
