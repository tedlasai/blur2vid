import os
import uuid
from pathlib import Path
import argparse

import gradio as gr
from PIL import Image
from diffusers.utils import export_to_video

from inference import load_model, inference_on_image

# -----------------------
# 1. Load model
# -----------------------
args = argparse.Namespace()
args.blur2vid_hf_repo_path = "tedlasai/blur2vid"
args.pretrained_model_path = "THUDM/CogVideoX-2b"
args.model_config_path = "training/configs/outsidephotos.yaml"
args.video_width = 1280
args.video_height = 720
args.seed = None

pipe, model_config = load_model(args)

OUTPUT_DIR = Path("/tmp/generated_videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_video_from_image(image: Image.Image, interval_key: str, num_inference_steps: int) -> str:
    """
    Wrapper for Gradio. Takes an image and returns a video path.
    """
    if image is None:
        raise gr.Error("Please upload an image first.")
    
    print("Generating video")
    import torch
    print("CUDA:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())

    args.num_inference_steps = num_inference_steps

    video_id = uuid.uuid4().hex
    output_path = OUTPUT_DIR / f"{video_id}.mp4"

    args.device = "cuda"

    pipe.to(args.device)
    processed_image, video = inference_on_image(pipe, image, interval_key, model_config, args)
    export_to_video(video, output_path, fps=20)

    if not os.path.exists(output_path):
        raise gr.Error("Video generation failed: output file not found.")

    return str(output_path)


with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown(
        """
        # 🖼️ ➜ 🎬 Recover Motion from a Blurry Image  
        
        This demo accompanies the paper **“Generating the Past, Present, and Future from a Motion-Blurred Image”**  
        by Tedla *et al.*, ACM Transactions on Graphics (SIGGRAPH Asia 2025).

        - 🌐 **Project page:** <https://blur2vid.github.io/>  
        - 💻 **Code:** <https://github.com/tedlasai/blur2vid/>  

        Upload a blurry image and the model will generate a short video showing the recovered motion based on your selection.
        Note: The image will be resized to 1280×720. We recommend uploading landscape-oriented images.
        """
    )

    with gr.Row():
        with gr.Column():
            image_in = gr.Image(
                type="pil",
                label="Input image",
                interactive=True,
            )

            with gr.Row():
                tense_choice = gr.Radio(
                    label="Select the interval to be generated:",
                    choices=["present", "past, present and future"],
                    value="past, present and future",
                    interactive=True,
                )

            num_inference_steps = gr.Slider(
                label="Number of inference steps",
                minimum=4,
                maximum=50,
                step=1,
                value=20,
                info="More steps = better quality but slower",
            )

            generate_btn = gr.Button("Generate video", variant="primary")

        with gr.Column():
            video_out = gr.Video(
                label="Generated video",
                format="mp4",
                autoplay=True,
                loop=True,
            )

    generate_btn.click(
        fn=generate_video_from_image,
        inputs=[image_in, tense_choice, num_inference_steps],
        outputs=video_out,
        api_name="predict",
    )

if __name__ == "__main__":
    demo.launch()
