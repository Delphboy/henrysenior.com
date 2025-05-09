---
title: "Superpixel Feature Extractor"
excerpt: "Generate features for superpixels and patches using pretrained models. Features are saves as `.npz` files."
collection: projects
github: https://github.com/Delphboy/superpixel-features
mathjax: true
math: true
---

Generate features for superpixels and patches using pretrained models. Features are saved as `.npz` files with keys for the features (`'feats'`), superpixel bounding box (`'bbox'`), and region adjacency edges (`'rag'`).

In terms of feature space, the code supports ResNet, CLIP, BLIP, and SigLIP features. Currently, only SLIC and Watershed superpixel segmentation algorithms are implemented (via scikit-image) however, patching is also supported (but not with the `--rag` flag).

To generate a collection of superpixels for the COCO dataset, see `runner.sh` for an example of how this can be achieved.

For compatiblity with the Karpathy Split of the COCO dataset, `merge_and_clean.py` is provided. This script will move and rename the superpixel feature files such that they can be used in place of the BUTD files in the original Karpathy Split JSON files. 

## Dependencies

```bash
conda env create -f environment.yml
```

```bash
conda create --name sp python=3.9
conda activate sp

python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install torch_geometric

python3 -m pip install git+https://github.com/openai/CLIP.git
python3 -m pip install salesforce-lavis

python3 -m pip install -U timm==1.0.14
python3 -m pip install -U transformers==4.48.1
```

## Parameters

| Name | Description |
|--|--|
| `--image_dir` | The directory containnig image inputs |
| `--save_dir` | The directory to save the `npz` files to |
| `--feature_extractor` | Which model to use? [BLIP / CLIP / RESNET / SIGLIP / VIT] |
| `--num_segments` | The number of superpixels to generate per image (Not compatible with `--whole_img`) |
| `--segmenter` | Which superpixel algorithm to use? [SLIC / WATERSHED] |
| `--whole_img` | (Flag) Generate a single feature for the whole image (Not compatible with `--rag`) |
| `--patches` | (Flag) Generate patch features instead of superpixel features (Not compatible with `--rag`) |
| `--rag` | (Flag) Generate the Region Adjacency Graph edges between superpixels |

> ⚠️
> The `--segmenter PATCHES` flag will try to generate the number of segments give in `--num-segments`. However, this number must result in
> the patching kernel size being a whole number. i.e. $$k = \sqrt{\frac{224 \times 224}{N}}$$ must be an integer (where $$N$$ is `--num_segments`). See
> [patcher.py](segmenters/patcher.py) for implementation details.


## Feature Sizes

| Model | Feature Dim |
|---|---|
| ResNet | 2048 |
| CLIP | 512 |
| BLIP | 768 |
| SigLIP | 1152 |
| ViT (`vit_b_16`) | 1000 |


## Examples

Generate 25 Watershed superpixel CLIP features for the Karpathy Test Set with RAG edges

```bash
python3 main.py --image_dir "/home/hsenior/coco/img/test2014/" \
    --save_dir "/home/hsenior/coco/superpixel_features/" \
    --model_id "CLIP" \
    --num_segments 25 \
    --segmenter "WATERSHED" \
    --rag
```

Generate whole image ResNet features for the Karpathy Validation set

```bash
python3 main.py --image_dir "/home/hsenior/coco/img/val2014/" \
    --save_dir "/home/hsenior/coco/superpixel_features/" \
    --model_id "RESNET" \
    --whole_img
```

Generate 196 patch BLIP features for the Karpathy Train Set

```bash
python3 main.py --image_dir "/home/hsenior/coco/img/train2014/" \
    --save_dir "/home/hsenior/coco/superpixel_features/" \
    --model_id "BLIP" \
    --num_segments 196 \
    --segmenter "PATCHER" \
```

