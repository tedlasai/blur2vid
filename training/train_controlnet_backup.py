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

import random
import signal
import sys
import threading
import time

import cv2
import yaml

sys.path.append('..')
import argparse
from PIL import Image
import logging
import math
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
import numpy as np
from decord import VideoReader
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

import diffusers
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.training_utils import (
    cast_training_params,
    free_memory,
)
from diffusers.utils import check_min_version, export_to_video, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

from controlnet_datasets import AblationFullMotionBlurDataset, AdobeMotionBlurDataset, FullMotionBlurDataset, GoPro2xMotionBlurDataset, GoProLargeMotionBlurDataset, OutsidePhotosDataset, GoProMotionBlurDataset, BAISTDataset, SimpleBAISTDataset
from controlnet_pipeline import ControlnetCogVideoXPipeline
from cogvideo_transformer import CogVideoXTransformer3DModel
from helpers import random_insert_latent_frame, transform_intervals
import os
import tempfile
from atomicwrites import atomic_write



if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)

def save_frames_as_pngs(video_array,output_dir, 
                        downsample_spatial=1,   # e.g. 2 to halve width & height
                        downsample_temporal=1): # e.g. 2 to keep every 2nd frame
    """
    Save each frame of a (T, H, W, C) numpy array as a PNG with no compression.
    """
    assert video_array.ndim == 4 and video_array.shape[-1] == 3, \
        "Expected (T, H, W, C=3) array"
    assert video_array.dtype == np.uint8, "Expected uint8 array"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # temporal downsample
    frames = video_array[::downsample_temporal]
    
    # compute spatially downsampled size
    T, H, W, _ = frames.shape
    new_size = (W // downsample_spatial, H // downsample_spatial)
    
    # PNG compression param: 0 = no compression
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
    
    for idx, frame in enumerate(frames):
        # frame is RGB; convert to BGR for OpenCV
        bgr = frame[..., ::-1]
        if downsample_spatial > 1:
            bgr = cv2.resize(bgr, new_size, interpolation=cv2.INTER_NEAREST)
        
        filename = os.path.join(output_dir, "frame_{:05d}.png".format(idx))
        success = cv2.imwrite(filename, bgr, png_params)
        if not success:
            raise RuntimeError("Failed to write frame ")


def get_args():
    parser = argparse.ArgumentParser(description="Training script for CogVideoX using config file.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the YAML config file."
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    args = argparse.Namespace(**config)

    # Convert nested config dict to an argparse.Namespace for easier downstream usage
    return args


# def read_video(video_path, start_index=0, frames_count=49, stride=1):
#     video_reader = VideoReader(video_path)
#     end_index = min(start_index + frames_count * stride, len(video_reader)) - 1
#     batch_index = np.linspace(start_index, end_index, frames_count, dtype=int)
#     numpy_video = video_reader.get_batch(batch_index).asnumpy()
#     return numpy_video
    

def log_validation(
    pipe,
    args,
    accelerator,
    pipeline_args,
    epoch,
    is_final_validation: bool = False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
    )
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    pipe = pipe.to(accelerator.device)
    # pipe.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    videos = []
    for _ in range(args.num_validation_videos):
        video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
        videos.append(video)

    free_memory()

    return videos


def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds


def compute_prompt_embeddings(
    tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim


        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy"]
    if args.optimizer not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not (args.optimizer.lower() not in ["adam", "adamw"]):
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if args.optimizer.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam


        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    return optimizer


def main(args):
    global signal_recieved_time
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Prepare models and scheduler
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.base_dir, args.pretrained_model_name_or_path), subfolder="tokenizer", revision=args.revision
    )

    text_encoder = T5EncoderModel.from_pretrained(
        os.path.join(args.base_dir, args.pretrained_model_name_or_path), subfolder="text_encoder", revision=args.revision
    )

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16

    ## TRYING NEW CONFIG LOADING
    config = CogVideoXTransformer3DModel.load_config(
    os.path.join(args.base_dir, args.pretrained_model_name_or_path),
    subfolder="transformer",
    revision=args.revision,
    variant=args.variant,
    )
    config["ablation_mode"] = args.ablation_mode if hasattr(args, "ablation_mode") else None 

     ##FINISH TRYING NEW CONFIG LOADING



    load_dtype = torch.bfloat16 if "5b" in os.path.join(args.base_dir, args.pretrained_model_name_or_path).lower() else torch.float16
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        os.path.join(args.base_dir, args.pretrained_model_name_or_path),
        subfolder="transformer",
        torch_dtype=load_dtype,
        ablation_mode=args.ablation_mode if hasattr(args, "ablation_mode") else None,
        revision=args.revision,
        variant=args.variant,
        low_cpu_mem_usage=False,
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        os.path.join(args.base_dir, args.pretrained_model_name_or_path), subfolder="vae", revision=args.revision, variant=args.variant
    )




    scheduler = CogVideoXDPMScheduler.from_pretrained(os.path.join(args.base_dir, args.pretrained_model_name_or_path), subfolder="scheduler")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    # We only train the additional adapter controlnet layers
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(True)
    vae.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.float16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters into fp32
        cast_training_params([transformer], dtype=torch.float32)

    trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    trainable_parameters_with_lr = {"params": trainable_parameters, "lr": args.learning_rate}
    params_to_optimize = [trainable_parameters_with_lr]

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    # Dataset and DataLoader
    if args.dataset == "adobe":
        train_dataset = AdobeMotionBlurDataset(
            data_dir=os.path.join(args.base_dir, args.video_root_dir),
            split = "train",
            image_size=(args.height, args.width), 
            stride=(args.stride_min, args.stride_max),
            sample_n_frames=args.max_num_frames,
            hflip_p=args.hflip_p,
        )
    elif args.dataset == "gopro":
        train_dataset = GoProMotionBlurDataset(
            data_dir=os.path.join(args.base_dir, args.video_root_dir),
            split = "train",
            image_size=(args.height, args.width), 
            stride=(args.stride_min, args.stride_max),
            sample_n_frames=args.max_num_frames,
            hflip_p=args.hflip_p,
        )
    elif args.dataset == "gopro2x":
        train_dataset = GoPro2xMotionBlurDataset(
            data_dir=os.path.join(args.base_dir, args.video_root_dir),
            split = "train",
            image_size=(args.height, args.width), 
            stride=(args.stride_min, args.stride_max),
            sample_n_frames=args.max_num_frames,
            hflip_p=args.hflip_p,
        )
    elif args.dataset == "goprolarge":
        train_dataset = GoProLargeMotionBlurDataset(
            data_dir=os.path.join(args.base_dir, args.video_root_dir),
            split = "train",
            image_size=(args.height, args.width), 
            stride=(args.stride_min, args.stride_max),
            sample_n_frames=args.max_num_frames,
            hflip_p=args.hflip_p,
        )
    elif args.dataset == "full":
        train_dataset = FullMotionBlurDataset(
            data_dir=os.path.join(args.base_dir, args.video_root_dir),
            split = "train",
            image_size=(args.height, args.width), 
            stride=(args.stride_min, args.stride_max),
            sample_n_frames=args.max_num_frames,
            hflip_p=args.hflip_p,
        )
    elif args.dataset == "fullablation":
        train_dataset = AblationFullMotionBlurDataset(
            data_dir=os.path.join(args.base_dir, args.video_root_dir),
            split = "train",
            image_size=(args.height, args.width), 
            stride=(args.stride_min, args.stride_max),
            sample_n_frames=args.max_num_frames,
            hflip_p=args.hflip_p,
            ablation_mode = args.ablation_mode, #this is not called for now
        )
    elif args.dataset == "baist":
        train_dataset = BAISTDataset(
            data_dir=os.path.join(args.base_dir, args.video_root_dir),
            split = "train",
            image_size=(args.height, args.width), 
            stride=(args.stride_min, args.stride_max),
            sample_n_frames=args.max_num_frames,
            hflip_p=args.hflip_p,
        ) #this is not called for now
    elif args.dataset == "simplebaist":
        train_dataset = SimpleBAISTDataset(
            data_dir=os.path.join(args.base_dir, args.video_root_dir),
            split = "train",
            image_size=(args.height, args.width), 
            stride=(args.stride_min, args.stride_max),
            sample_n_frames=args.max_num_frames,
            hflip_p=args.hflip_p,
        )
        

    if args.dataset == "adobe":
        val_dataset = AdobeMotionBlurDataset(
            data_dir=os.path.join(args.base_dir, args.video_root_dir),
            split = args.val_split,
            image_size=(args.height, args.width), 
            stride=(args.stride_min, args.stride_max),
            sample_n_frames=args.max_num_frames,
            hflip_p=args.hflip_p,
        )
    elif args.dataset == "outsidephotos":
        
        val_dataset = OutsidePhotosDataset(
            data_dir=os.path.join(args.base_dir, args.video_root_dir),
            image_size=(args.height, args.width), 
            stride=(args.stride_min, args.stride_max),
            sample_n_frames=args.max_num_frames,
            hflip_p=args.hflip_p,
        )
        train_dataset = val_dataset #dummy dataset
    elif args.dataset == "gopro":
        val_dataset = GoProMotionBlurDataset(
            data_dir=os.path.join(args.base_dir, args.video_root_dir),
            split = args.val_split,
            image_size=(args.height, args.width), 
            stride=(args.stride_min, args.stride_max),
            sample_n_frames=args.max_num_frames,
            hflip_p=args.hflip_p,
        )
    elif args.dataset == "gopro2x":
        val_dataset = GoPro2xMotionBlurDataset(
            data_dir=os.path.join(args.base_dir, args.video_root_dir),
            split = args.val_split,
            image_size=(args.height, args.width), 
            stride=(args.stride_min, args.stride_max),
            sample_n_frames=args.max_num_frames,
            hflip_p=args.hflip_p,
        )
    elif args.dataset == "goprolarge":
        val_dataset = GoProLargeMotionBlurDataset(
            data_dir=os.path.join(args.base_dir, args.video_root_dir),
            split = args.val_split,
            image_size=(args.height, args.width), 
            stride=(args.stride_min, args.stride_max),
            sample_n_frames=args.max_num_frames,
            hflip_p=args.hflip_p,
        )
    elif args.dataset == "full":
        val_dataset = FullMotionBlurDataset(
            data_dir=os.path.join(args.base_dir, args.video_root_dir),
            split = args.val_split,
            image_size=(args.height, args.width), 
            stride=(args.stride_min, args.stride_max),
            sample_n_frames=args.max_num_frames,
            hflip_p=args.hflip_p,
        )
    elif args.dataset == "fullablation":
        val_dataset = AblationFullMotionBlurDataset(
            data_dir=os.path.join(args.base_dir, args.video_root_dir),
            split = args.val_split,
            image_size=(args.height, args.width), 
            stride=(args.stride_min, args.stride_max),
            sample_n_frames=args.max_num_frames,
            hflip_p=args.hflip_p,
            ablation_mode = args.ablation_mode, #this is not called for now
        )
    elif args.dataset == "baist":
        val_dataset = BAISTDataset(
            data_dir=os.path.join(args.base_dir, args.video_root_dir),
            split = args.val_split,
            image_size=(args.height, args.width), 
            stride=(args.stride_min, args.stride_max),
            sample_n_frames=args.max_num_frames,
            hflip_p=args.hflip_p,
        )
    elif args.dataset == "simplebaist":
        val_dataset = SimpleBAISTDataset(
            data_dir=os.path.join(args.base_dir, args.video_root_dir),
            split = args.val_split,
            image_size=(args.height, args.width), 
            stride=(args.stride_min, args.stride_max),
            sample_n_frames=args.max_num_frames,
            hflip_p=args.hflip_p,
        )
        
    def encode_video(video):
        video = video.to(accelerator.device, dtype=vae.dtype)
        video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        latent_dist = vae.encode(video).latent_dist.sample() * vae.config.scaling_factor
        return latent_dist.permute(0, 2, 1, 3, 4).to(memory_format=torch.contiguous_format)
    
    # def atomic_save(save_path, accelerator):

    #     dir_name = os.path.dirname(save_path)
    #     with tempfile.NamedTemporaryFile(delete=False, dir=dir_name) as tmp_file:
    #         tmp_path = tmp_file.name
    #         # Close the file so that it can be moved later
    #     #delete anything at the tmp_path
    #     if accelerator.is_main_process:
    #         accelerator.save_state(tmp_path) #just a backup incase things go crazy
    #         accelerator.save_state(save_path)
    #     os.remove(tmp_path)  
    #     accelerator.wait_for_everyone()



    def atomic_save(save_path, accelerator):
        parent = os.path.dirname(save_path)
        tmp_dir = tempfile.mkdtemp(dir=parent)
        backup_dir = save_path + "_backup"

        try:
            # Save state into the temp directory
            accelerator.save_state(tmp_dir)

            # Backup existing save_path if it exists
            if os.path.exists(save_path):
                os.rename(save_path, backup_dir)

            # Atomically move temp directory into place
            os.rename(tmp_dir, save_path)

            # Clean up the backup directory
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)

        except Exception as e:
            # Clean up temp directory on failure
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

            # Restore from backup if replacement failed
            if os.path.exists(backup_dir):
                if os.path.exists(save_path):
                    shutil.rmtree(save_path)
                os.rename(backup_dir, save_path)

            raise e

        

    def collate_fn(examples):
        blur_img = [example["blur_img"] for example in examples]
        videos = [example["video"] for example in examples]
        if "high_fps_video" in examples[0]:
            high_fps_videos = [example["high_fps_video"] for example in examples]
            high_fps_videos = torch.stack(high_fps_videos)
            high_fps_videos = high_fps_videos.to(memory_format=torch.contiguous_format).float()
        if "bbx" in examples[0]:
            bbx = [example["bbx"] for example in examples]
            bbx = torch.stack(bbx)
            bbx = bbx.to(memory_format=torch.contiguous_format).float()
        prompts = [example["caption"] for example in examples]
        file_names = [example["file_name"] for example in examples]
        num_frames = [example["num_frames"] for example in examples]
        # if full_file_names in examples[0]:
        #     full_file_names = [example["full_file_name"] for example in examples]
        input_intervals = [example["input_interval"] for example in examples]
        output_intervals = [example["output_interval"] for example in examples]
        ablation_condition = [example["ablation_condition"] for example in examples] if "ablation_condition" in examples[0] else None


        videos = torch.stack(videos)
        videos = videos.to(memory_format=torch.contiguous_format).float()

        blur_img = torch.stack(blur_img)
        blur_img = blur_img.to(memory_format=torch.contiguous_format).float()


        input_intervals = torch.stack(input_intervals)
        if args.dataset == "gopro":
            input_intervals = input_intervals.to(memory_format=torch.contiguous_format).long()  #this is a bug, but I trained it like this on GOPRO (sets intervals all to 0), model doesn't need intervals for this dataset cause its always 7 frames in the same spacing
        else:
            input_intervals = input_intervals.to(memory_format=torch.contiguous_format).float()

        output_intervals = torch.stack(output_intervals)
        if args.dataset == "gopro":
            output_intervals = output_intervals.to(memory_format=torch.contiguous_format).long()  #this is a bug, but I trained it like this on GOPRO (sets intervals all to 0), model doesn't need intervals for this dataset cause its always 7 frames in the same spacing
        else:
            output_intervals = output_intervals.to(memory_format=torch.contiguous_format).float()

        #just used for ablation studies
        ablation_condition = torch.stack(ablation_condition) if ablation_condition is not None else None
        if ablation_condition is not None:
            ablation_condition = ablation_condition.to(memory_format=torch.contiguous_format).float()

        out_dict = {
            "file_names": file_names,
            "blur_img": blur_img,
            "videos": videos,
            "num_frames": num_frames,
            "prompts": prompts,
            "input_intervals": input_intervals,
            "output_intervals": output_intervals,
        }

        if "high_fps_video" in examples[0]:
            out_dict["high_fps_video"] = high_fps_videos
        if "bbx" in examples[0]:
            out_dict["bbx"] = bbx
        if ablation_condition is not None:
            out_dict["ablation_condition"] = ablation_condition
        return out_dict

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler, val_dataloader = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler, val_dataloader
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "cogvideox-controlnet"
        accelerator.init_trackers(tracker_name, config=vars(args))

    
    accelerator.register_for_checkpointing(transformer, optimizer, lr_scheduler)
    save_path = os.path.join(args.output_dir, f"checkpoint")

    #check if the checkpoint already exists
    if os.path.exists(save_path):
        accelerator.load_state(save_path)
        logger.info(f"Loaded state from {save_path}")



    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            if not args.just_validate:
                models_to_accumulate = [transformer]
                with accelerator.accumulate(models_to_accumulate):
                    model_input = encode_video(batch["videos"]).to(dtype=weight_dtype)  # [B, F, C, H, W]
                    prompts = batch["prompts"]
                    image_latent = encode_video(batch["blur_img"]).to(dtype=weight_dtype)  # [B, F, C, H, W]
                    input_intervals = batch["input_intervals"]
                    output_intervals = batch["output_intervals"] 
                    ablation_condition = batch["ablation_condition"] if "ablation_condition" in batch else None

                    batch_size = len(prompts)
                    # True = use real prompt (conditional); False = drop to empty (unconditional)
                    guidance_mask = torch.rand(batch_size, device=accelerator.device) >= 0.2  

                    # build a new prompts list: keep the original where mask True, else blank
                    per_sample_prompts = [
                        prompts[i] if guidance_mask[i] else ""
                        for i in range(batch_size)
                    ]
                    prompts = per_sample_prompts

                    # encode prompts
                    prompt_embeds = compute_prompt_embeddings(
                        tokenizer,
                        text_encoder,
                        prompts,
                        model_config.max_text_seq_length,
                        accelerator.device,
                        weight_dtype,
                        requires_grad=False,
                    )

                    # Sample noise that will be added to the latents
                    noise = torch.randn_like(model_input)
                    batch_size, num_frames, num_channels, height, width = model_input.shape

                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, scheduler.config.num_train_timesteps, (batch_size,), device=model_input.device
                    )
                    timesteps = timesteps.long()
            
                    # Prepare rotary embeds
                    image_rotary_emb = (
                        prepare_rotary_positional_embeddings(
                            height=args.height,
                            width=args.width,
                            num_frames=num_frames,
                            vae_scale_factor_spatial=vae_scale_factor_spatial,
                            patch_size=model_config.patch_size,
                            attention_head_dim=model_config.attention_head_dim,
                            device=accelerator.device,
                        )
                        if model_config.use_rotary_positional_embeddings
                        else None
                    )

                    # Add noise to the model input according to the noise magnitude at each timestep (this is the forward diffusion process)
                    noisy_model_input = scheduler.add_noise(model_input, noise, timesteps)

                    input_intervals = transform_intervals(input_intervals, frames_per_latent=4)
                    output_intervals = transform_intervals(output_intervals, frames_per_latent=4)
                    
                    #first interval is always rep
                    noisy_model_input, target, condition_mask, intervals = random_insert_latent_frame(image_latent, noisy_model_input, model_input, input_intervals, output_intervals, special_info=args.special_info)
                    
                    for i in range(batch_size):
                        if not guidance_mask[i]:
                            noisy_model_input[i][condition_mask[i]] = 0

                    # Predict the noise residual
                    model_output = transformer(
                        hidden_states=noisy_model_input,
                        encoder_hidden_states=prompt_embeds,
                        intervals=intervals,
                        condition_mask=condition_mask,
                        timestep=timesteps,
                        image_rotary_emb=image_rotary_emb,
                        return_dict=False,
                        ablation_condition = ablation_condition
                    )[0]

                    #this line below is also scaling the input which is bad - so the model is also learning to scale this input latent somehow
                    #thus, we need to replace the first frame with the original frame later
                    model_pred = scheduler.get_velocity(model_output, noisy_model_input, timesteps)



                    alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                    weights = 1 / (1 - alphas_cumprod)
                    while len(weights.shape) < len(model_pred.shape):
                        weights = weights.unsqueeze(-1)



                    loss = torch.mean((weights * (model_pred[~condition_mask] - target[~condition_mask]) ** 2).reshape(batch_size, -1), dim=1)
                    loss = loss.mean()
                    accelerator.backward(loss)

                    if accelerator.state.deepspeed_plugin is None:
                        if not args.just_validate:
                            optimizer.step()
                        optimizer.zero_grad()

                    lr_scheduler.step()

            
                    #wait for all processes to finish
                    accelerator.wait_for_everyone()

                    
                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1

                        if signal_recieved_time != 0:
                            if time.time() - signal_recieved_time > 60:
                                print("Signal received, saving state and exiting")
                                #accelerator.save_state(save_path)
                                atomic_save(save_path, accelerator)
                                signal_recieved_time = 0
                                exit(0)
                            else:
                                exit(0)

                        if accelerator.is_main_process:
                            if global_step % args.checkpointing_steps == 0:
                                #accelerator.save_state(save_path)
                                atomic_save(save_path, accelerator)
                                logger.info(f"Saved state to {save_path}")

                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                    if global_step >= args.max_train_steps:
                        break

            print("Step", step)
            accelerator.wait_for_everyone()
            
            if step == 0 or args.validation_prompt is not None and (step + 1) % args.validation_steps == 0:
                # Create pipeline
                pipe = ControlnetCogVideoXPipeline.from_pretrained(
                    os.path.join(args.base_dir, args.pretrained_model_name_or_path),
                    transformer=unwrap_model(transformer),
                    text_encoder=unwrap_model(text_encoder),
                    vae=unwrap_model(vae),
                    scheduler=scheduler,
                    torch_dtype=weight_dtype,
                )

                print("Length of validation dataset: ", len(val_dataloader))
                #create a pipeline per accelerator device (for faster inference)
                with torch.autocast(str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"):
                    for batch in val_dataloader:
                        frame = ((batch["blur_img"][0].permute(0,2,3,1).cpu().numpy() + 1)*127.5).astype(np.uint8)
                        pipeline_args = {
                            "prompt": "",
                            "negative_prompt": "",
                            "image": frame,
                            "input_intervals": batch["input_intervals"][0:1],
                            "output_intervals": batch["output_intervals"][0:1],
                            "ablation_condition": batch["ablation_condition"][0:1] if "ablation_condition" in batch else None,
                            "guidance_scale": args.guidance_scale,
                            "use_dynamic_cfg": args.use_dynamic_cfg,
                            "height": args.height,
                            "width": args.width,
                            "num_frames": args.max_num_frames,
                            "num_inference_steps": args.num_inference_steps,
                        }

                        modified_filenames = []
                        filenames = batch['file_names']
                        for file in filenames:
                            modified_filenames.append(os.path.splitext(file)[0] + ".mp4")

                        num_frames = batch["num_frames"][0]
                        #save the gt_video output
                        if args.dataset not in ["outsidephotos"]:
                            gt_video = batch["videos"][0].permute(0,2,3,1).cpu().numpy()
                            gt_video = ((gt_video + 1) * 127.5)/255
                            gt_video = gt_video[0:num_frames]
                            
                            for file in modified_filenames:
                                #create the directory if it does not exist
                                gt_file_name = os.path.join(args.output_dir, "gt", modified_filenames[0])
                                os.makedirs(os.path.dirname(gt_file_name), exist_ok=True)
                                if args.dataset in ["baist", "simplebaist"]:
                                    bbox = batch["bbx"][0].cpu().numpy().astype(np.int32)
                                    gt_video = gt_video[:, bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                                    gt_video = np.array([cv2.resize(frame, (160, 192)) for frame in gt_video])

                                save_frames_as_pngs((gt_video*255).astype(np.uint8), gt_file_name.replace(".mp4", "").replace("gt", "gt_frames"))
                                export_to_video(gt_video, gt_file_name, fps=20)


                                if "high_fps_video" in batch:
                                    high_fps_video = batch["high_fps_video"][0].permute(0,2,3,1).cpu().numpy()
                                    high_fps_video = ((high_fps_video + 1) * 127.5)/255
                                    gt_file_name = os.path.join(args.output_dir, "gt_highfps", modified_filenames[0])
                                    
                        
                        if args.dataset in ["adobe", "full", "baist", "outsidephotos", "gopro2x", "goprolarge", "simplebaist"]:
                            for file in modified_filenames:
                                #create the directory if it does not exist
                                blurry_file_name = os.path.join(args.output_dir, "blurry", modified_filenames[0].replace(".mp4", ".png"))
                                #save the blurry image
                                os.makedirs(os.path.dirname(blurry_file_name), exist_ok=True)
                                if args.dataset in ["baist", "simplebaist"]:
                                    bbox = batch["bbx"][0].cpu().numpy().astype(np.int32)
                                    frame0 = frame[0][bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                                    #resize to  192x160
                                    frame0 = cv2.resize(frame0, (160, 192)) 
                                    Image.fromarray(frame0).save(blurry_file_name)
                                else:
                                    Image.fromarray(frame[0]).save(blurry_file_name)

                        videos = log_validation(
                            pipe=pipe,
                            args=args,
                            accelerator=accelerator,
                            pipeline_args=pipeline_args,
                            epoch=epoch,
                        )

                        for i, video in enumerate(videos):
                            prompt = (
                                pipeline_args["prompt"][:25]
                                .replace(" ", "_")
                                .replace(" ", "_")
                                .replace("'", "_")
                                .replace('"', "_")
                                .replace("/", "_")
                            )
                            video = video[0:num_frames]
                            filename = os.path.join(args.output_dir, "deblurred", modified_filenames[0])
                            print("Deblurred file name", filename)
                            os.makedirs(os.path.dirname(filename), exist_ok=True)
                            if args.dataset in ["baist", "simplebaist"]:
                                bbox = batch["bbx"][0].cpu().numpy().astype(np.int32)
                                video = video[:, bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                                #resize to  192x160
                                video = np.array([cv2.resize(frame, (160, 192)) for frame in video])
                            save_frames_as_pngs((video*255).astype(np.uint8), filename.replace(".mp4", "").replace("deblurred", "deblurred_frames"))
                            export_to_video(video, filename, fps=20)

                            accelerator.wait_for_everyone()

                if args.just_validate:
                    exit(0)

    accelerator.wait_for_everyone()
    accelerator.end_training()

signal_recieved_time = 0

def handle_signal(signum, frame):
    global signal_recieved_time
    signal_recieved_time = time.time()

    print(f"Signal {signum} received at {time.ctime()}")

    with open("/datasets/sai/gencam/cogvideox/interrupted.txt", "w") as f:
        f.write(f"Training was interrupted at {time.ctime()}")

if __name__ == "__main__":

    args = get_args()

    print("Registering signal handler")
    #Register the signal handler (catch SIGUSR1)
    signal.signal(signal.SIGUSR1, handle_signal)

    main_thread = threading.Thread(target=main, args=(args,))
    main_thread.start()

    print("SIGNAL RECIEVED TIME", signal_recieved_time)
    while signal_recieved_time!= 0:
        time.sleep(1)
    
    #call main with args as a thread

