<div align="center">

# FaceMoE : Mixture of Experts for <br> Low-Resolution Face Recognition
<h3><strong>ECCV 2026</strong></h3>

[Kartik Narayan](https://kartik-3004.github.io/portfolio/) &emsp; [Vishal M. Patel](https://engineering.jhu.edu/faculty/vishal-patel/)  

Johns Hopkins University

<a href='https://kartik-3004.github.io/FaceMoE/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href=''><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://huggingface.co/kartiknarayan/FaceMoE'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange'></a>

</div>

Official implementation of **[FaceMoE : Mixture of Experts for Low-Resolution Face Recognition](https://kartik-3004.github.io/portfolio/papers/FaceMOE.pdf)**.
<hr />


This repository contains the FaceMoE training and evaluation code (Mixture-of-Experts models built on Swin-style backbones) used for experiments on low-resolution face recognition. The repo includes training scripts, configs, validation/evaluation utilities, and example shell helpers for multi-/single-GPU runs.

## Contents

- `train.py`, `train_fc.py`, `train_full.py` - main training entrypoints (distributed-aware).
- `train.sh`, `test.sh` - example commands for common runs.
- `configs/` - Python config files (e.g. `wf4m.py`, `tinyface_ft_fc.py`, `briar_ft_full.py`).
- `validation_*` - evaluation/validation scripts for different datasets (IJB, HQ, LQ, ...).
- `weights/`, `pretrained_weights/` - example model weight folders (not all present by default).
- `requirements.txt`, `environment.yml` - python dependencies / conda environment.

## Quick start

1. Create an environment and install dependencies.
   - Create the conda environment provided:

	   conda env create -f environment.yml
	   conda activate face_moe

2. Edit a config in `configs/` or use one of the provided configs. Important fields in each config include: dataset paths (`rec`), output directory (`output`), batch size, learning rate, number of experts (`num_experts`) and top-k routing (`k`).

3. Training examples

   The training scripts are written to work with PyTorch distributed launch (torchrun). Below are example commands used in this repository.

   - Multi-GPU training (8 GPUs):
	```	
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py configs/wf4m.py
	```
 
	 Or to run the FC or full training variants (examples already used in experiments):
	```	
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_fc.py configs/tinyface_ft_fc.py
	```
 	```	
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_full.py configs/tinyface_ft_full.py
  	```	

   - Single-GPU debugging / small runs (not distributed):
	```	
	CUDA_VISIBLE_DEVICES=0 python train.py configs/wf4m.py
	```
 
   Notes:
   - The training scripts read a single positional argument: the path to a Python config in `configs/`.
   - The training code uses PyTorch distributed internals. For single-process runs the code falls back to a local process group.

4. Evaluation / testing examples

   Several evaluation scripts are provided under `validation_*`. Example commands from `test.sh` (adjust paths and flags for your environment):

   - HQ validation example:
	```	
   CUDA_VISIBLE_DEVICES=0 python validation_hq/validate_hq.py \
	   --model_load_path /path/to/FaceMoE/weights/your_model/model.pt \
	   --model_type swin_moe \
	   --batch_size 1400 \
	   --num_experts 3 \
	   --k 2 \
	   --image_size 120
 	```

   - Low-resolution / tinyface validation example:
	```	
   CUDA_VISIBLE_DEVICES=0 python validation_lq/validate_tinyface.py \
	   --model_load_path /path/to/FaceMoE/weights/your_model/model.pt \
	   --model_type swin_moe \
	   --batch_size 1200 \
	   --num_experts 3 \
	   --k 2 \
	   --image_size 120
	```
 
   - IJB evaluation example (IJBB/IJBC):
	```	
   CUDA_VISIBLE_DEVICES=0 python validation_ijb/eval_ijb.py \
	   --model_load_path /path/to/FaceMoE/weights/your_model/model.pt \
	   --data_root /path/to/data/ijb/IJBB/ \
	   --model_type swin_moe \
	   --batch-size 1024 \
	   --num_experts 3 \
	   --k 2 \
	   --target IJBB
	```
 
   Notes:
   - Each validation script exposes flags for the model path, data root, batch size, num_experts, k (top-k routing), and image size.

## Configs

Look in `configs/` for ready-to-run examples. Common configs found in this repository include:

- `wf4m.py` — example WF configuration.
- `tinyface_ft_fc.py`, `tinyface_ft_full.py` — tinyface fine-tuning configs.
- `briar_ft_fc.py`, `briar_ft_full.py` — briar fine-tuning configs.

Edit these files to point `rec`/dataset paths and set `output` to where checkpoints and logs will be written.

## Checkpoints & outputs

During training the scripts save per-rank checkpoints like `checkpoint_gpu_{rank}.pt` in the config `output` directory and a model file `model.pt` (rank 0). Keep an eye on disk usage as some experiments produce large checkpoints.

## Notes and tips

- If you plan to log to Weights & Biases, set the fields in the config (e.g. `wandb_key`, `wandb_entity`) and enable the related section.
- For large distributed runs ensure `WORLD_SIZE` / GPUs and NCCL network are properly configured for your cluster.

## Where to look next

- Training entrypoint: `train.py` (distributed training loop, config loader, schedulers and partial-FC integration).
- Data loader: `dataset.py` (dataset & augmentations).
- Loss and partial-FC: `losses.py`, `partial_fc_v2.py`.
- Validation scripts: `validation_hq/`, `validation_lq/`, `validation_ijb/`.

## Reproducing a typical run (example)

1. Prepare dataset(s) and update a config file in `configs/`.
2. Create environment and install requirements.
3. Launch training on 8 GPUs (example):
	```	
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py configs/wf4m.py
 	```	

4. After a model is saved, validate with a validation script (adjust model and data paths):

   CUDA_VISIBLE_DEVICES=0 python validation_hq/validate_hq.py --model_load_path /path/to/checkpoint/model.pt ...

## Contact / License

See `LICENSE` for license details. If you have questions about the code, run commands, or configs, contact: knaraya4@jhu.edu
