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

import signal
import sys
import threading
import time
import cv2
sys.path.append('..')
from PIL import Image
import logging
import math
import os
from pathlib import Path

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from transformers import AutoTokenizer, T5EncoderModel

import diffusers
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    free_memory,
)
from diffusers.utils import check_min_version, export_to_video, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

from controlnet_datasets import FullMotionBlurDataset, GoPro2xMotionBlurDataset, OutsidePhotosDataset, GoProMotionBlurDataset, BAISTDataset
from controlnet_pipeline import ControlnetCogVideoXPipeline
from cogvideo_transformer import CogVideoXTransformer3DModel
from helpers import random_insert_latent_frame, transform_intervals
import os
from utils import save_frames_as_pngs, compute_prompt_embeddings, prepare_rotary_positional_embeddings, encode_prompt, get_optimizer, atomic_save, get_args
if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)


def log_validation(
    pipe,
    args,
    accelerator,
    pipeline_args,
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

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    videos = []
    for _ in range(args.num_validation_videos):
        video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
        videos.append(video)

    free_memory() #delete the pipeline to free up memory

    return videos



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
    config = CogVideoXTransformer3DModel.load_config(
    os.path.join(args.base_dir, args.pretrained_model_name_or_path),
    subfolder="transformer",
    revision=args.revision,
    variant=args.variant,
    )

    load_dtype = torch.bfloat16 if "5b" in os.path.join(args.base_dir, args.pretrained_model_name_or_path).lower() else torch.float16
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        os.path.join(args.base_dir, args.pretrained_model_name_or_path),
        subfolder="transformer",
        torch_dtype=load_dtype,
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
    DATASET_REGISTRY = {
        "gopro": GoProMotionBlurDataset,
        "gopro2x": GoPro2xMotionBlurDataset,
        "full": FullMotionBlurDataset,
        "baist": BAISTDataset,
        "outsidephotos": OutsidePhotosDataset,  # val-only special (no split)
    }

    if args.dataset not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_dataset_class = DATASET_REGISTRY[args.dataset]
    val_dataset_class   = train_dataset_class

    common_kwargs = dict(
        data_dir=os.path.join(args.base_dir, args.video_root_dir),
        output_dir = args.output_dir,
        image_size=(args.height, args.width),
        stride=(args.stride_min, args.stride_max),
        sample_n_frames=args.max_num_frames,
        hflip_p=args.hflip_p,
    )

    def build_kwargs(is_train: bool):
        """Return constructor kwargs, adding split"""
        kw = dict(common_kwargs)
        kw["split"] = "train" if is_train else args.val_split
        return kw

    train_dataset = train_dataset_class(**build_kwargs(is_train=True))
    val_dataset   = val_dataset_class(**build_kwargs(is_train=False))
            
    def encode_video(video):
        video = video.to(accelerator.device, dtype=vae.dtype)
        video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        latent_dist = vae.encode(video).latent_dist.sample() * vae.config.scaling_factor
        return latent_dist.permute(0, 2, 1, 3, 4).to(memory_format=torch.contiguous_format)
        
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
        input_intervals = [example["input_interval"] for example in examples]
        output_intervals = [example["output_interval"] for example in examples]

        videos = torch.stack(videos)
        videos = videos.to(memory_format=torch.contiguous_format).float()

        blur_img = torch.stack(blur_img)
        blur_img = blur_img.to(memory_format=torch.contiguous_format).float()


        input_intervals = torch.stack(input_intervals)
        input_intervals = input_intervals.to(memory_format=torch.contiguous_format).float()

        output_intervals = torch.stack(output_intervals)
        output_intervals = output_intervals.to(memory_format=torch.contiguous_format).float()


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
                                atomic_save(save_path, accelerator)
                                signal_recieved_time = 0
                                exit(0)
                            else:
                                exit(0)

                        if accelerator.is_main_process:
                            if global_step % args.checkpointing_steps == 0:
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
                                gt_file_name = os.path.join(args.output_dir, "gt", modified_filenames[0])
                                os.makedirs(os.path.dirname(gt_file_name), exist_ok=True)
                                if args.dataset == "baist":
                                    bbox = batch["bbx"][0].cpu().numpy().astype(np.int32)
                                    gt_video = gt_video[:, bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                                    gt_video = np.array([cv2.resize(frame, (160, 192)) for frame in gt_video]) #resize to 192x160

                                save_frames_as_pngs((gt_video*255).astype(np.uint8), gt_file_name.replace(".mp4", "").replace("gt", "gt_frames"))
                                export_to_video(gt_video, gt_file_name, fps=20)


                                if "high_fps_video" in batch:
                                    high_fps_video = batch["high_fps_video"][0].permute(0,2,3,1).cpu().numpy()
                                    high_fps_video = ((high_fps_video + 1) * 127.5)/255
                                    gt_file_name = os.path.join(args.output_dir, "gt_highfps", modified_filenames[0])
                                    
                        
                        #save the blurred image
                        if args.dataset in ["full", "outsidephotos", "gopro2x", "baist"]:
                            for file in modified_filenames:
                                blurry_file_name = os.path.join(args.output_dir, "blurry", modified_filenames[0].replace(".mp4", ".png"))
                                os.makedirs(os.path.dirname(blurry_file_name), exist_ok=True)
                                if args.dataset == "baist":
                                    bbox = batch["bbx"][0].cpu().numpy().astype(np.int32)
                                    frame0 = frame[0][bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                                    frame0 = cv2.resize(frame0, (160, 192))  #resize to 192x160
                                    Image.fromarray(frame0).save(blurry_file_name)
                                else:
                                    Image.fromarray(frame[0]).save(blurry_file_name)

                        videos = log_validation(
                            pipe=pipe,
                            args=args,
                            accelerator=accelerator,
                            pipeline_args=pipeline_args
                        )

                        #save the output video frames as pngs (uncompressed results) and mp4 (compressed results easily viewable)
                        for i, video in enumerate(videos):
                            video = video[0:num_frames]
                            filename = os.path.join(args.output_dir, "deblurred", modified_filenames[0])
                            os.makedirs(os.path.dirname(filename), exist_ok=True)
                            if args.dataset == "baist":
                                bbox = batch["bbx"][0].cpu().numpy().astype(np.int32)
                                video = video[:, bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                                video = np.array([cv2.resize(frame, (160, 192)) for frame in video]) #resize to 192x160
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

    while signal_recieved_time!= 0:
        time.sleep(1)
    
    #call main with args as a thread

