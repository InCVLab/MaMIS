# InCVLab

**MaMIS** is a research codebase for **3D medical image generation**, latent-space diffusion modeling, and downstream conditional synthesis tasks. The repository supports latent diffusion generation and ControlNet-guided synthesis for anatomical or lesion-conditioned image generation.

The full source code will be provided once the paper is accepted.

---

## Highlights
- Latent diffusion models for random 3D medical image generation.
- ControlNet-based conditional generation using mask or structural guidance.
- MONAI-based preprocessing and sliding-window inference for large 3D volumes.
- Docker-based setup for reproducible GPU environments.

---

## Repository Overview

```text
CVLab_Med/
├── configs/                  # Model and experiment configuration files
├── eval/                     # Evaluation metrics
├── nets/                     # Model structures
├── train/                    # Training scripts for diffusion and ControlNet
├── models/                   # Pretrained or trained model checkpoints
├── inference_diffusion.py    # Latent diffusion inference
├── inference_controlnet.py   # ControlNet-based inference
├── docker-compose.yml        # Docker Compose configuration
├── Dockerfile                # Docker environment definition
└── requirements.txt          # Python package requirements
```

---

## System Requirements

The codebase was developed and tested with the following environment:

| Component | Recommended version |
|---|---|
| Python | >= 3.10 |
| PyTorch | 2.8 |
| CUDA | 12.8 |
| MONAI | Compatible with the installed PyTorch version |

For Mamba-based models, `causal-conv1d` and `mamba-ssm` should be built against the same CUDA/PyTorch environment used for training and inference.

---

## Installation

Create and activate your Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Install `causal-conv1d` and `mamba-ssm` from source-compatible builds:

```bash
python -m pip install --no-cache-dir --no-build-isolation --no-binary=causal-conv1d causal-conv1d==1.6.1

python -m pip install --no-cache-dir --no-build-isolation --no-binary=mamba-ssm mamba-ssm==2.3.1
```

---

## Docker Installation

Docker is recommended for reproducible GPU experiments.

Before building the container:

1. Update the PyTorch base image in the `Dockerfile` if needed.
2. Set `TORCH_CUDA_ARCH_LIST` according to your GPU architecture.
3. Update the dataset mount path in `docker-compose.yml`, especially the `/workspaces/data` volume.

Build and launch the container:

```bash
docker compose build --no-cache
docker compose up -d
docker exec -it cvlab_med bash
```

---

## Pretrained Models

The `VAE` checkpoint can be downloaded directly from the NVIDIA project:

[autoencoder_v2](https://huggingface.co/nvidia/NV-Generate-MR/tree/main/models)

Pretrained `diffusion` and `controlnet` checkpoints can be downloaded from:

[Google Drive checkpoint folder](https://drive.google.com/drive/folders/13HbNrBMjmKk2qUOcnuLXJbH-WyH6INv8?usp=drive_link)

Place downloaded checkpoints under the `models/` directory, following the expected structure:

```text
models/
├── mamba_diffusion/
└── mamba_controlnet/
```

---

# 1. Diffusion

The diffusion module performs random generation in the learned VAE latent space.

## Configuration

Update the model configuration, class, and body_region in:

```text
configs/config_model.json
```

Supported diffusion model names include:

```text
unet_diffusion
mamba_diffusion
```

---

## Diffusion Training

The diffusion model is trained on latent embeddings extracted from the VAE encoder.

`--meta_path` should point to a JSONL metadata file containing the training samples.

Use `--multi_res`, `--bucket_batch_size`, and `--bucket_batch_mode` for multi-resolution training.

Example command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node 4 --nnodes 1 --master_port 29510 -m train.train_diffusion --num_gpus 4 \
--model_name mamba_diffusion \
--model_suffix 4_multi_res_all_modals \
--meta_path jsonl_path \
--multi_res true \
--bucket_batch_size 64x64x64:8,128x128x16:8,128x128x32:4,128x128x64:2 \
--amp true \
--amp_dtype bfloat16 \
--resume true \
--pretrained_path ckpt_path
```

---

## Diffusion Inference

Diffusion inference generates synthetic latent embeddings, which are decoded into 3D images by the pretrained VAE decoder.

Example command:

```bash
CUDA_VISIBLE_DEVICES=0 python inference_diffusion.py \
  --diff_model_name mamba_diffusion \
  --config_model_path /workspaces/src/configs/config_model.json \
  --input_path jsonl_path \
  --output_dir dir_path \
  --diffusion_pretrained_path ckpt_path \
  --vae_pretrained_path ckpt_path \
  --input_shape 4 128 128 16 \
  --num_inference_steps 100
```

---

# 2. ControlNet

The ControlNet module performs conditional 3D medical image generation using structural inputs such as masks.

## Configuration

Update the ControlNet configuration, class, and body_region in:

```text
configs/config_model.json
```

Supported ControlNet model names include:

```text
unet_controlnet
mamba_controlnet
```

---

## ControlNet Training

Trained checkpoints are saved under:

```text
models/{model_name}/
```

Use `--pixdim_map` for multi-class training. Spacing is used in transforms to normalize all labels to the same spacing as the latent embeddings.

The classes must have the same number of IDs in the label.

Example command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node 4 --nnodes 1 --master_port 29510 -m train.train_controlnet --num_gpus 4 \
--model_name mamba_controlnet \
--diffusion_name mamba_diffusion \
--model_suffix inhouse_4_51251264 \
--batch_size 4 \
--output_size 512 512 64 \
--meta_path jsonl_path \
--pixdim_map spinal_core=-1,-1,1.9 ankle=-1,-1,1.4 \
--resume false

```

---

## ControlNet Inference

Before running inference, create a mask folder and place the input condition masks inside it or provide a JSONL path to `mask_dir`.

Example command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node 4 --nnodes 1 --master_port 29510 -m inference_controlnet --num_gpus 4 \
--diff_model_name mamba_diffusion \
--controlnet_model_name mamba_controlnet \
--config_model_path /workspaces/src/configs/config_model.json \
--mask_dir dir_path \
--output_dir dir_path \
--diffusion_pretrained_path ckpt_path \
--controlnet_pretrained_path ckpt_path \
--vae_pretrained_path ckpt_path \
--input_shape 4 128 128 16 \
--modality mri \
--body_region ankle \
--pixdim -1 -1 1.4 \
--num_inference_steps 30 \
--decode_overlap 0.4 \
--save_as_original_size true
```

---

# Data Format

## Image folders

For folder-based inference, place NIfTI images in the input directory:

```text
input_path/
├── case_001.nii.gz
├── case_002.nii.gz
└── case_003.nii.gz
```

## JSONL metadata

For training or batch inference, each line of the JSONL file should describe one sample. An example is:

```json
{"image": "/path/to/image.nii.gz", "modality": "mri", "body_region": "ankle", "spacing": [1.0, 1.0, 1.0], "spatial_shape": [128, 128, 16, 4], "fold": 1}
```

For conditional or multi-domain training, metadata may additionally include:

```json
{"image": "/path/to/image.nii.gz", "label": "/path/to/label.nii.gz", "modality": "mri", "body_region": "ankle", "spacing": [1.0, 1.0, 1.0], "spatial_shape": [128, 128, 16, 4], "fold": 1}
```

---

# Evaluation

For random generation without paired ground truth, recommended metrics include:

- FID in a medical feature space
- pairwise diversity in feature or latent space
- downstream task performance, when applicable

---

# Reproducibility Notes

For reproducible training and inference:

1. Record the exact checkpoint path used for each experiment.
2. Save the full configuration file from `configs/config_model.json`.
3. Add/Modify classes and body_regions in `configs/config_model.json` before training and inference.
4. Keep preprocessing parameters fixed, especially `pixdim`, `spatial_size`, and `image_size`.
5. Use the same VAE checkpoint for latent extraction and decoding.
6. Report GPU type, CUDA version, PyTorch version, and MONAI version.

---

# Citation

If you use this repository in academic work, please cite the associated manuscript or project once available.

```bibtex
@misc{cvlab_med,
  title        = {MaMIS: Mamba-Enhanced ControlNet for Medical Image Synthesis},
  author       = {CVLab},
  year         = {2026},
  howpublished = {GitHub repository}
}
```

