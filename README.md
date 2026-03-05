# Right for the Right Regions (R4RR)

This repository packages the full experiment stack used for R4RR:
- R4RR training and sweeps
- baseline models (Vanilla, Upweight, ABN, GALS variants, AFR, CLIP baselines)
- teacher map generation (WeCLIP+ and GALS attention pipelines)
- one-command reproduction wrappers for each dataset

The code is organized so you can either:
1. run full paper-facing reproductions from `pipelines/train_CNN`, or
2. run individual methods directly from `repro_runs`.

## Repository Layout

```text
RightForTheRightRegions/
├── README.md
├── READMEGals.md
├── requirements_runs.txt
├── requirements_weclip.txt
├── configs/
│   ├── waterbirds95_optimized_hparams.yaml
│   ├── waterbirds100_optimized_hparams.yaml
│   ├── redmeat_optimized_hparams.yaml
│   ├── decoymnist_hparams.yaml
│   └── r4rr_optimized_hparams.yaml
├── data/
│   └── make_decoymnist_pngs.py
├── pipelines/
│   ├── generate_r4rr_maps/
│   ├── generate_gals_maps/
│   └── train_CNN/
├── repro_runs/
│   ├── r4rr/
│   ├── other_models/
│   └── third_party/
│       ├── GALS/
│       ├── MakeMNIST/
│       ├── afr/
│       └── group_DRO/
└── WeCLIPPlus/
```

## Environment Setup

You will typically use **two environments**:
- `runs` env: model training / sweeps / reproduction wrappers
- `weclip` env: WeCLIP+ teacher-map generation

### 1) Runs environment

```bash
conda create -n r4rr-runs python=3.10 -y
conda activate r4rr-runs

# Install torch/torchvision first for your CUDA, then:
pip install -r requirements_runs.txt
```

### 2) WeCLIP+ environment

`WeCLIPPlus` dependencies are older and are usually most stable in Python 3.8.

```bash
conda create -n r4rr-weclip python=3.8 -y
conda activate r4rr-weclip
pip install -r requirements_weclip.txt
```

## Dataset Setup

The repo expects datasets under `data/` (or explicit paths passed to scripts).

### Waterbirds-95

Download from the official Group DRO release:
- https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz

Example:

```bash
mkdir -p data
cd data
wget https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz
tar -xzf waterbird_complete95_forest2water2.tar.gz
```

Expected folder:

```text
data/waterbird_complete95_forest2water2/
```

### Waterbirds-100

This split should be reconstructed using the Group DRO workflow.

Use the vendored Group DRO docs and script:
- `repro_runs/third_party/group_DRO/README.md`
- `repro_runs/third_party/group_DRO/dataset_scripts/generate_waterbirds.py`

Important notes:
- `generate_waterbirds.py` uses hardcoded path variables at the top of the file.
- You need CUB + Places assets as described in Group DRO docs.
- The generated folder should be named/placed as:

```text
data/waterbird_1.0_forest2water2/
```

### RedMeat (Food-101 subset)

Food-101 download (official):
- https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/

Training/eval scripts expect a RedMeat dataset directory containing `all_images.csv` (and associated image paths). Typical layout used in this repo:

```text
data/food-101-redmeat/
├── all_images.csv
└── split_images/
   ├── train/<class>/*.jpg
   └── val/<class>/*.jpg
```

### DecoyMNIST PNG conversion

Generate DecoyMNIST arrays + PNG layout via:

```bash
python data/make_decoymnist_pngs.py
```

This runs MakeMNIST's `00_make_data.py` and writes:

```text
repro_runs/third_party/MakeMNIST/data/DecoyMNIST_png/
├── train/<digit>/*.png
└── test/<digit>/*.png
```

---

## Generate Teacher/Attention Maps

## A) R4RR teacher maps (WeCLIP+)

Scripts:
- `pipelines/generate_r4rr_maps/generate_pseudo_masks_waterbirds.py`
- `pipelines/generate_r4rr_maps/generate_pseudo_masks_redmeat.py`
- `pipelines/generate_r4rr_maps/generate_pseudo_masks_DecoyMNIST.py`

### Path compatibility note

These scripts currently resolve WeCLIPPlus as `<repo-root>/code/WeCLIPPlus`.
If your checkout is `RightForTheRightRegions/WeCLIPPlus`, add this one-time symlink:

```bash
cd /path/to/RightForTheRightRegions
mkdir -p code
ln -sfn "$PWD/WeCLIPPlus" code/WeCLIPPlus
```

### Waterbirds teacher maps

```bash
conda activate r4rr-weclip
python pipelines/generate_r4rr_maps/generate_pseudo_masks_waterbirds.py \
  --repo-root "$PWD" \
  --src-img-dir "$PWD/data/waterbird_complete95_forest2water2" \
  --class-name bird \
  --setup-data \
  --results-dir results_waterbirds95_r4rr
```

Run again with `--src-img-dir "$PWD/data/waterbird_1.0_forest2water2"` for WB100.

### RedMeat teacher maps

```bash
conda activate r4rr-weclip
python pipelines/generate_r4rr_maps/generate_pseudo_masks_redmeat.py \
  --repo-root "$PWD" \
  --split-images-dir "$PWD/data/food-101-redmeat/split_images" \
  --class-name meat \
  --setup-data \
  --results-dir results_redmeat_r4rr
```

### DecoyMNIST teacher maps (WeCLIP+ path)

```bash
conda activate r4rr-weclip
python pipelines/generate_r4rr_maps/generate_pseudo_masks_DecoyMNIST.py \
  --repo-root "$PWD" \
  --class-name digit \
  --results-dir results_decoy_r4rr
```

Note: this Decoy WeCLIP+ script assumes VOC-style `JPEGImages` / `ImageSets` are already prepared for WeCLIPPlus.

### Switching CLIP/DINO backbones in WeCLIP+

Use script flags to swap teacher backbones:
- `--clip-backend` (`openai`, `openclip`, `siglip2` depending on script)
- `--clip-model`
- `--clip-pretrained`
- `--dino-model`
- `--dino-fts-dim`
- `--dino-decoder-layers`

Example (RedMeat):

```bash
python pipelines/generate_r4rr_maps/generate_pseudo_masks_redmeat.py \
  --repo-root "$PWD" \
  --split-images-dir "$PWD/data/food-101-redmeat/split_images" \
  --class-name meat \
  --setup-data \
  --clip-backend openclip \
  --clip-model ViT-B-16 \
  --clip-pretrained laion2b_s34b_b88k \
  --dino-model xcit_medium_24_p16 \
  --dino-fts-dim 512 \
  --dino-decoder-layers 2 \
  --results-dir results_redmeat_openclip_xcit
```

## B) GALS attention maps

Scripts:
- `pipelines/generate_gals_maps/run_generate_waterbirds_rn50_attentions_95_100_debug.py`
- `pipelines/generate_gals_maps/run_generate_waterbirds_vit_attentions_95_100.py`
- `pipelines/generate_gals_maps/run_generate_redmeat_rn50_attentions.py`
- `pipelines/generate_gals_maps/run_generate_redmeat_vit_attentions.py`
- `pipelines/generate_gals_maps/run_generate_decoymnist_rn50_attentions.py`

### Waterbirds (RN50 + ViT)

```bash
conda activate r4rr-runs
python pipelines/generate_gals_maps/run_generate_waterbirds_rn50_attentions_95_100_debug.py
python pipelines/generate_gals_maps/run_generate_waterbirds_vit_attentions_95_100.py
```

### RedMeat (RN50 + ViT)

```bash
conda activate r4rr-runs
python pipelines/generate_gals_maps/run_generate_redmeat_rn50_attentions.py \
  --dataset-dir food-101-redmeat

python pipelines/generate_gals_maps/run_generate_redmeat_vit_attentions.py \
  --dataset-dir food-101-redmeat
```

### DecoyMNIST (RN50 Grad-CAM)

```bash
conda activate r4rr-runs
python pipelines/generate_gals_maps/run_generate_decoymnist_rn50_attentions.py \
  --png-root "$PWD/repro_runs/third_party/MakeMNIST/data/DecoyMNIST_png" \
  --output-root "$PWD/repro_runs/third_party/MakeMNIST/data/DecoyMNIST_png/clip_rn50_attention_gradcam"
```

---

## One-Command Reproduction Runs

These wrappers run each method once per dataset, using hyperparameters from `configs/*_optimized_hparams.yaml`, and write consolidated outputs.

Scripts:
- `pipelines/train_CNN/recreate_waterbirds95_runs.py`
- `pipelines/train_CNN/recreate_waterbirds100_runs.py`
- `pipelines/train_CNN/recreate_redmeat_runs.py`
- `pipelines/train_CNN/recreate_decoymnist_runs.py`

All wrappers produce:

```text
logs/recreate/<dataset>_<timestamp>/
├── <method>/stdout.log
├── summary.csv
└── summary.json
```

### Waterbirds-95

```bash
conda activate r4rr-runs
python pipelines/train_CNN/recreate_waterbirds95_runs.py \
  --gals-our-masks-path <PATH_TO_GALS_OUR_MASKS_DIR> \
  --r4rr-rn50-map-path <PATH_TO_R4RR_RN50_MAPS> \
  --r4rr-vit-map-path <PATH_TO_R4RR_VIT_MAPS> \
  --r4rr-optimized-map-path <PATH_TO_FINAL_R4RR_MAPS> \
  --r4rr-ablation-map-path <PATH_TO_R4RR_ABLATION_MAPS>
```

### Waterbirds-100

```bash
conda activate r4rr-runs
python pipelines/train_CNN/recreate_waterbirds100_runs.py \
  --gals-our-masks-path <PATH_TO_GALS_OUR_MASKS_DIR> \
  --r4rr-rn50-map-path <PATH_TO_R4RR_RN50_MAPS> \
  --r4rr-vit-map-path <PATH_TO_R4RR_VIT_MAPS> \
  --r4rr-optimized-map-path <PATH_TO_FINAL_R4RR_MAPS> \
  --r4rr-ablation-map-path <PATH_TO_R4RR_ABLATION_MAPS>
```

### RedMeat

```bash
conda activate r4rr-runs
python pipelines/train_CNN/recreate_redmeat_runs.py \
  --dataset-path "$PWD/data/food-101-redmeat" \
  --gals-our-masks-path <PATH_TO_GALS_OUR_MASKS_DIR> \
  --r4rr-rn50-map-path <PATH_TO_R4RR_RN50_MAPS> \
  --r4rr-vit-map-path <PATH_TO_R4RR_VIT_MAPS> \
  --r4rr-optimized-map-path <PATH_TO_FINAL_R4RR_MAPS>
```

### DecoyMNIST

```bash
conda activate r4rr-runs
python pipelines/train_CNN/recreate_decoymnist_runs.py \
  --png-root "$PWD/repro_runs/third_party/MakeMNIST/data/DecoyMNIST_png" \
  --teacher-map-path "$PWD/repro_runs/third_party/MakeMNIST/data/DecoyMNIST_png/clip_rn50_attention_gradcam"
```

Tip: add `--dry-run` to any wrapper to print commands without executing training.

---

## Running Individual Methods Directly

For method-level runs, go to `repro_runs/`.

## R4RR

### Train
- `repro_runs/r4rr/train/r4rr_waterbirds.py`
- `repro_runs/r4rr/train/r4rr_redmeat.py`
- `repro_runs/r4rr/train/r4rr_decoy_fixed.py`

Examples:

```bash
python repro_runs/r4rr/train/r4rr_waterbirds.py \
  "$PWD/data/waterbird_complete95_forest2water2" \
  <TEACHER_MAP_PATH> \
  --attention_epoch 109 --kl_lambda 295.30 \
  --base_lr 4.82e-5 --classifier_lr 2.93e-3 --lr2_mult 0.409
```

```bash
python repro_runs/r4rr/train/r4rr_redmeat.py \
  "$PWD/data/food-101-redmeat" \
  <TEACHER_MAP_PATH> \
  --attention-epoch 2 --kl-lambda 11.44 \
  --base_lr 2.40e-3 --classifier_lr 2.33e-4 --lr2-mult 1.567
```

```bash
python repro_runs/r4rr/train/r4rr_decoy_fixed.py \
  --png-root "$PWD/repro_runs/third_party/MakeMNIST/data/DecoyMNIST_png" \
  --teacher-map-path "$PWD/repro_runs/third_party/MakeMNIST/data/DecoyMNIST_png/clip_rn50_attention_gradcam" \
  --attention-epoch 7 --kl-lambda 495.61 --lr 0.001 --epochs 19
```

### Sweeps / ablations
- `repro_runs/r4rr/sweeps/r4rr_waterbirds_sweep.py`
- `repro_runs/r4rr/sweeps/r4rr_redmeat_sweep.py`
- `repro_runs/r4rr/ablations/r4rr_waterbirds_invert.py`
- `repro_runs/r4rr/ablations/r4rr_waterbirds_joint.py`

## Other models

### Waterbirds
- `repro_runs/other_models/waterbirds/sweeps/gals_waterbirds_sweep.py`
- `repro_runs/other_models/waterbirds/sweeps/clip_lr_waterbirds_sweep.py`
- `repro_runs/other_models/waterbirds/sweeps/afr_waterbirds_sweep.py`
- `repro_runs/other_models/waterbirds/baselines/clip_zeroshot_waterbirds.py`

### RedMeat
- `repro_runs/other_models/redmeat/sweeps/gals_redmeat_sweep.py`
- `repro_runs/other_models/redmeat/sweeps/clip_lr_redmeat_sweep.py`
- `repro_runs/other_models/redmeat/sweeps/afr_redmeat_sweep.py`
- `repro_runs/other_models/redmeat/baselines/clip_zeroshot_redmeat.py`

### DecoyMNIST
- `repro_runs/other_models/decoymnist/train/upweight_decoy_fixed.py`
- `repro_runs/other_models/decoymnist/train/abn_decoy_fixed.py`
- `repro_runs/other_models/decoymnist/train/gals_decoy_fixed.py`
- `repro_runs/other_models/decoymnist/train/afr_decoy_fixed.py`
- `repro_runs/other_models/decoymnist/baselines/clip_lr_decoy_fixed.py`
- `repro_runs/other_models/decoymnist/baselines/clip_zeroshot_decoy.py`

---

## Hyperparameter Config Files

The canonical hparam files are:
- `configs/waterbirds95_optimized_hparams.yaml`
- `configs/waterbirds100_optimized_hparams.yaml`
- `configs/redmeat_optimized_hparams.yaml`
- `configs/decoymnist_hparams.yaml`
- `configs/r4rr_optimized_hparams.yaml`

The `pipelines/train_CNN/recreate_*` wrappers read these YAMLs directly.

---

## Practical Notes / Troubleshooting

- Use absolute paths for dataset and map folders when possible.
- If a method is missing required map paths in the wrapper scripts, it is marked as `skipped` in summary outputs.
- For CLIP+LR stability on some systems, prefer conservative solver settings (`l2:lbfgs`).
- For DecoyMNIST, avoid very high dataloader worker counts.
- For newer GPUs, verify your PyTorch build supports the GPU architecture before long runs.

---

## Citations / Upstream Dependencies

This repo vendors and builds on:
- GALS: `repro_runs/third_party/GALS`
- Group DRO: `repro_runs/third_party/group_DRO`
- AFR: `repro_runs/third_party/afr`
- CDEP: `repro_runs/third_party/CDEP`
- WeCLIPPlus: `WeCLIPPlus`

Please cite original works when using their components.

## Acknowledgements

We thank the authors and maintainers of **WeCLIP+**, **GALS**, **GroupDRO**, **AFR**, and **CDEP** for their foundational contributions. A substantial portion of this repository builds directly on their open-source code, ideas, and released tooling.
