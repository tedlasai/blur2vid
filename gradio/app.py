import os
import uuid
from pathlib import Path

import gradio as gr
from PIL import Image

# -----------------------
# 1. Load your model here
# -----------------------
# Example:
# from my_model_lib import MyVideoModel
# model = MyVideoModel.from_pretrained("your/model/hub/id")

OUTPUT_DIR = Path("/tmp/generated_videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_video_from_image(image: Image.Image) -> str:
    video_id = uuid.uuid4().hex
    output_path = OUTPUT_DIR / f"{video_id}.mp4"

    # 1. Preprocess image
    # img_tensor = preprocess(image)  # your code

    # 2. Run model
    # frames = model(img_tensor)  # e.g. np.ndarray of shape (T, H, W, 3), dtype=uint8

    # 3. Save frames to video
    # iio.imwrite(
    #     uri=output_path,
    #     image=frames,
    #     fps=16,
    #     codec="h264",
    # )

    return str(output_path)


def demo_predict(image: Image.Image) -> str:
    """
    Wrapper for Gradio. Takes an image and returns a video path.
    """
    if image is None:
        raise gr.Error("Please upload an image first.")

    video_path = generate_video_from_image(image)
    if not os.path.exists(video_path):
        raise gr.Error("Video generation failed: output file not found.")
    return video_path


with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown(
        """
        # 🖼️ ➜ 🎬 Recover motion from a blurry image!

        Upload an image and the model will generate a short video.
        """
    )

    with gr.Row():
        with gr.Column():
            image_in = gr.Image(
                type="pil",
                label="Input image",
                interactive=True,
            )
            generate_btn = gr.Button("Generate video", variant="primary")
        with gr.Column():
            video_out = gr.Video(
                label="Generated video",
                format="mp4",  # ensures browser-friendly output
                autoplay=True,
            )

    generate_btn.click(
        fn=demo_predict,
        inputs=image_in,
        outputs=video_out,
        api_name="predict",
    )

if __name__ == "__main__":
    demo.launch()
