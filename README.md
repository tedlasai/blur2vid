# Generating the Past, Present, and Future from a Motion-Blurred Image

**SaiKiran Tedla, Kelly Zhu, Trevor Canham, Felix Taubner, Michael Brown, Kiriakos Kutulakos, David Lindell**  
University of Toronto & York University  

### 🤗 [Demo](https://huggingface.co/spaces/tedlasai/blur2vid)
### 📄 [Paper (PDF)](https://dl.acm.org/doi/10.1145/3763306)  
### 🌐 [Project Page](https://blur2vid.github.io)  
### 📂 [Checkpoints](https://huggingface.co/tedlasai/blur2vid/tree/main)
---

## 📌 Citation

If you use our dataset, code, or model, please cite:

```bibtex
@article{Tedla2025Blur2Vid,
  title        = {Generating the Past, Present, and Future from a Motion-Blurred Image},
  author       = {Tedla, SaiKiran and Zhu, Kelly and Canham, Trevor and Taubner, Felix and Brown, Michael and Kutulakos, Kiriakos and Lindell, David},
  journal      = {ACM Transactions on Graphics},
  year         = {2025},
  note         = {SIGGRAPH Asia.}
}
```

---

## 🚀 Getting Started

This guide explains how to train and evaluate our **video diffusion model** for video from a single motion-blurred image.

---

### 🔧 Environment Setup

```bash
conda env create -f setup/environment.yml
conda activate blur2vid
```

- Install PyTorch and all dependencies listed in the YAML file.   

---



### 🧪 Testing (In-the-Wild)

Running the model on images is as simple as:
```bash
conda activate blur2vid

python inference.py --image_path assets/dummy_image.png --output_path output/
```

We also provide an interactive Gradio demo:

```bash
python gradio/app.py
```

---


### 🧪 Testing (GOPRO/BAIST)
To test on these datasets, please use `configs/gopro_test.yaml`, `configs/gopro_2x_test.yaml`, or `configs/baist_test.yaml`, depending on the experiment you are interested in.

Set the following paths in your YAML config (feel free to change others paths to match your configuration):
1. Set the basedir in the corresponding yaml file in `training/configs/` to the path of the repository. This will be the directory that contains the README.md. 
2. Download checkpoints with ``python setup/download_checkpoints.py baist`` or ``python setup/download_checkpoints.py gopro``, respectively. The checkpoint directory should appear in the ``training`` directory.
3. Put the GOPRO or BAIST dataset in `datasets/baist` or `datasets/GOPRO_7` respectively.

```bash
cd training
conda activate blur2vid
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3 #4 GPUS

accelerate launch --config_file accelerator_configs/accelerator_val_config.yaml --multi_gpu \
  train_controlnet.py \
  --config "SPECIFIED CONFIG FILE"
```

---

### 🏋️‍♂️ Training

To train our model.
1. You can run `python setup/download_cogvideo_weights.py`, you should have a folder named `CogVideoX-2b` at the project root containing the Cogvideox model weights. This gives you the initial weights.
2. Setup dataset structure as mentioned below. We sourced our data from 5 highspeed video datasets.

```bash
cd training
conda activate blur2vid
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3 #4 GPUS

accelerate launch --config_file accelerator_configs/accelerator_val_config.yaml --multi_gpu \
  train_controlnet.py \
  --config "configs/full_train.yaml"
```

### Dataset Structure

Each dataset is organized in a consistent folder structure under `datasets/FullDataset/`. 

#### Directory Layout
```
datasets/FullDataset/
└── {DatasetName}/
    ├── lower_fps_frames/
    │   └── {SceneName}/
    │       └── *.png
    ├── train.txt
    └── test.txt
```

#### Example

For the Adobe240 dataset:
```
datasets/FullDataset/Adobe240/
├── lower_fps_frames/
│   ├── GOPR9633/
│   │   └── frame_0001.png, frame_0002.png, ...
│   ├── GOPR9634/
│   └── ...
├── train.txt
└── test.txt
```

## Components

- **lower_fps_frames/**: Contains video sequences organized by scene, with each scene stored as a sequence of PNG frames
- **train.txt**: List of training sequences
- **test.txt**: List of test sequences

---


### 📜 Notes

- Checkpoints are available on the [project page](https://blur2vid.github.io).  
- We utilize `extra/moMets-parallel-gopro.py` and `extra/moMets-parallel-baist.py` to compute all metrics for this project.

---

### 📨 Contact

For questions or issues, please reach out through the [project page](https://blur2vid.github.io) or contact [Sai Tedla](mailto:tedlasai@gmail.com).


#### References and Additional Details
Hugginface version I used to borrow pipeline - https://github.com/huggingface/diffusers/tree/92933ec36a13989981a6fc4189857e8b4dc2c38d 

CogvideoX Controlnet Extention - https://github.com/user-attachments/assets/d3cd3cc4-de95-453f-bbf7-ccbe1711fc3c

Original code and models [CogVideoX](https://github.com/THUDM/CogVideo/tree/main).  
