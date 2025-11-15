from huggingface_hub import HfApi
import os
token = os.getenv("HF_TOKEN")
api = HfApi(token=token)
# ---- REQUIRED ----
repo_id = "tedlasai/blur2vid"   # example: "sai/blur2vid"

files_to_delete = [
    # "cogvideox-baist-test.zip",
    # "cogvideox-gopro-test.zip",
    # "cogvideox-gopro-2x-test.zip",
    # "cogvideox-full-test.zip",
    # "cogvideox-outsidephotos.zip",
    "README.md"
]

# -------------------

for file in files_to_delete:
    api.delete_file(
        path_in_repo=file,
        repo_id=repo_id,
        token=token
    )
    print(f"Deleted: {file}")

print("Done!")
