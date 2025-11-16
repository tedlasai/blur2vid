# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from pathlib import Path
import io
import yaml

from PIL import Image, ImageCms
import torch
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel
from safetensors.torch import load_file
import diffusers
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler
from diffusers.utils import check_min_version, export_to_video
from huggingface_hub import hf_hub_download

from controlnet_pipeline import ControlnetCogVideoXPipeline
from cogvideo_transformer import CogVideoXTransformer3DModel

from training.utils import save_frames_as_pngs
from training.helpers import get_conditioning

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")


def convert_to_srgb(img: Image):
    if 'icc_profile' in img.info:
        icc = img.info['icc_profile']
        src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        dst_profile = ImageCms.createProfile("sRGB")
        img = ImageCms.profileToProfile(img, src_profile, dst_profile, outputMode='RGB')
    else:
        img = img.convert("RGB")  # Assume sRGB
    return img


INTERVALS = {
    "present": {
        "in_start": 0, 
        "in_end": 16, 
        "out_start": 0, 
        "out_end": 16, 
        "center": 8, 
        "window_size": 16, 
        "mode": "1x", 
        "fps": 240
    },
    "past_present_and_future": {
        "in_start": 4, 
        "in_end": 12, 
        "out_start": 0, 
        "out_end": 16, 
        "center": 8, 
        "window_size": 16,
        "mode": "2x", 
        "fps": 240,
    },
}


def convert_to_batch(
    image, 
    interval_key="present", 
    image_size=(720, 1280),
):
    interval = INTERVALS[interval_key]

    inp_int, out_int, num_frames = get_conditioning(
        in_start=interval['in_start'],
        in_end=interval['in_end'],
        out_start=interval['out_start'],
        out_end=interval['out_end'],
        mode=interval['mode'],
        fps=interval['fps'],
    )

    blur_img_original = convert_to_srgb(image)
    H, W = blur_img_original.size

    blur_img = blur_img_original.resize((image_size[1], image_size[0])) # pil is width, height
    blur_img = torch.from_numpy(np.array(blur_img)[None]).permute(0, 3, 1, 2).contiguous().float()
    blur_img = blur_img / 127.5 - 1.0

    data = {
        "original_size": (H, W),
        'blur_img': blur_img,
        'caption': "",
        'input_interval': inp_int,
        'output_interval': out_int,
        'height': image_size[0],
        'width': image_size[1],
        'num_frames': num_frames,
    }
    return data


def load_model(args):
    with open(args.model_config_path) as f:
        model_config = yaml.safe_load(f)

    load_dtype = torch.float16
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_path, 
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=model_config["revision"],
        variant=model_config["variant"],
        low_cpu_mem_usage=False,
    )
    weight_path = hf_hub_download(
        repo_id=args.blur2vid_hf_repo_path, 
        filename="cogvideox-outsidephotos/checkpoint/model.safetensors"
    )
    transformer.load_state_dict(load_file(weight_path))

    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_path, 
        subfolder="text_encoder", 
        revision=model_config["revision"],
    )

    tokenizer = T5Tokenizer.from_pretrained(
        args.pretrained_model_path, 
        subfolder="tokenizer", 
        revision=model_config["revision"],
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_path, 
        subfolder="vae", 
        revision=model_config["revision"],
        variant=model_config["variant"],
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(
        args.pretrained_model_path, 
        subfolder="scheduler"
    )

    # Enable slicing or tiling if VRAM is low!
    vae.enable_slicing()
    vae.enable_tiling()

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.bfloat16

    text_encoder.to(dtype=weight_dtype)
    transformer.to(dtype=weight_dtype)
    vae.to(dtype=weight_dtype)

    pipe = ControlnetCogVideoXPipeline.from_pretrained(
        args.pretrained_model_path,
        tokenizer=tokenizer,
        transformer=transformer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=weight_dtype,
    )
    
    scheduler_args = {}

    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)

    return pipe, model_config


def inference_on_image(pipe, image, interval_key, model_config, args):
    # If passed along, set the training seed now.
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # run inference
    generator = torch.Generator(device=args.device).manual_seed(args.seed) if args.seed else None

    with torch.autocast(args.device, enabled=True):
        batch = convert_to_batch(image, interval_key, (args.video_height, args.video_width))

        frame = batch["blur_img"].permute(0, 2, 3, 1).cpu().numpy()
        frame = (frame + 1.0) * 127.5
        frame = frame.astype(np.uint8)
        pipeline_args = {
            "prompt": "",
            "negative_prompt": "",
            "image": frame,
            "input_intervals": torch.stack([batch["input_interval"]]).float(),
            "output_intervals": torch.stack([batch["output_interval"]]).float(),
            "guidance_scale": model_config["guidance_scale"],
            "use_dynamic_cfg": model_config["use_dynamic_cfg"],
            "height": batch["height"],
            "width": batch["width"],
            "num_frames": torch.tensor([[model_config["max_num_frames"]]]), # torch.tensor([[batch["num_frames"]]]),
            "num_inference_steps": args.num_inference_steps,
        }

        input_image = frame

        num_frames = batch["num_frames"]  # this is the actual number of frames, the video generation is padded by one frame

        print(f"Running inference for interval {interval_key}...")
        video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]

        video = video[0:num_frames]

    return input_image, video


def main(args):
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True)

    image_path = Path(args.image_path)

    is_dir = image_path.is_dir()

    if is_dir:
        image_paths = sorted(list(image_path.glob("*.*")))
    else:
        image_paths = [image_path]

    pipe, model_config = load_model(args)
    pipe.to(args.device)

    for image_path in image_paths:
        image = Image.open(image_path)

        processed_image, video = inference_on_image(pipe, image, "past_present_and_future", model_config, args)

        vid_output_path = output_path / f"{image_path.stem}.mp4"
        export_to_video(video, vid_output_path, fps=20)
        
        # save input image as well
        inpug_image_output_path = output_path / f"{image_path.stem}_input.png"
        Image.fromarray(processed_image[0]).save(inpug_image_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to image input or directory containing input images",
    )
    parser.add_argument(
        "--blur2vid_hf_repo_path",
        type=str,
        default="tedlasai/blur2vid",
        help="hf repo containing the weight files",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="THUDM/CogVideoX-2b",
        help="repo id or path for pretrained CogVideoX model",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="training/configs/outsidephotos.yaml",
        help="path to model config yaml",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/",
        help="path to output",
    )
    parser.add_argument(
        "--video_width",
        type=int,
        default=1280,
        help="video resolution width",
    )
    parser.add_argument(
        "--video_height",
        type=int,
        default=720,
        help="video resolution height",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="number of DDIM steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random generator seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="inference device",
    )
    args = parser.parse_args()
    main(args)

# python inference.py --image_path assets/dummy_image.png --output_path output/
