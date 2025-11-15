from huggingface_hub import HfApi
import os
#run with HF_TOKEN = your_hf_token before python_command
api = HfApi(token=os.getenv("HF_TOKEN"))
folders = ["/datasets/sai/blur2vid/training/cogvideox-baist-test", 
           "/datasets/sai/blur2vid/training/cogvideox-gopro-test",
           "/datasets/sai/blur2vid/training/cogvideox-gopro-2x-test",
           "/datasets/sai/blur2vid/training/cogvideox-full-test",
           "/datasets/sai/blur2vid/training/cogvideox-outsidephotos"]
for folder in folders:
    api.upload_folder(
        folder_path=folder,
        repo_id="tedlasai/blur2vid",
        repo_type="model",
        path_in_repo=os.path.basename(folder)
    )