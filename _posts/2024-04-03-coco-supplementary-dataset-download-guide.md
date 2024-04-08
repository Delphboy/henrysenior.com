---
title: COCO Supplementary Dataset Download Guide
collection: words
permalink: /words/2024-04-03-coco-supplementary-dataset-download-guide
excerpt: "A guide to downloading the BUTD and VSUA/SGAE supplementary datasets for COCO."
---

**Warning:** The directories and script calls may not work with simply copy and paste due to scripts being directories etc. Do not copy and paste blindly.

**Note:** The directory `/data/scratch/$USER/` is used as an example and may not work on your machine. It is the QMUL HPC scratch directory.

Since the [Bottom Up Top Down paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Anderson_Bottom-Up_and_Top-Down_CVPR_2018_paper.html), most paper in image captioning no longer take the raw images as input. Instead, they take the object detection features from this paper (or some other representation of the images) as input. Here is your guide to setting up the data.

All the scripts required to extract and setup the dataset(s) can be found [here](https://github.com/Delphboy/coco-scripts).

There is a complete setup script the QMUL HPC system available [here](https://github.com/Delphboy/coco-scripts/blob/main/full_install.qsub). Copy the contents of this file into a new job script and run it. **If you only want to BUTD data, remove the commands below this `######`**

Run the download and run the QMUL setup script with:

```bash
cd
wget -c https://raw.githubusercontent.com/Delphboy/coco-scripts/main/full_install.qsub -O coco.qsub
qsub coco.qsub -N "COCO-setup"
```
## Directory Creation
1. You're going to need to set up an empty directory where you can set everything up ie) `mkdir /data/scratch/$USER/COCO/`

## Karpathy Splits
As the COCO test server is a thing of the past, [Karpathy proposed an alternative split for the COCO dataset](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Karpathy_Deep_Visual-Semantic_Alignments_2015_CVPR_paper.pdf) which is $113, 287$ / $5000$ / $5000$ train/val/test. I keep a copy of the [JSON files on my GitHub](https://github.com/Delphboy/karpathy-splits)

```bash
cd /data/scratch/$USER/COCO/

wget -c https://github.com/Delphboy/karpathy-splits/raw/main/dataset_coco.json?download= -O dataset_coco.json
```

You will then need to run the following [prepro_labels.py](https://github.com/Delphboy/meshed-memory-transformer/blob/master/scripts/prepro_labels.py) script. There is a good chance you will have this script in a codebase under the `scripts/` directory.

```bash
python3 prepro_labels.py --input_json /data/scratch/$USER/COCO/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```

## Bottom Up Top Down Object Detection

Most of the time you will have to download the BUTD data
```bash
mkdir /data/scratch/$USER/COCO/bu_data
cd /data/scratch/$USER/COCO/bu_data
wget https://storage.googleapis.com/bottom-up-attention/trainval.zip
unzip trainval.zip
```

However, it's pre-downloaded on the QMUL HPC system
```bash
mkdir /data/scratch/$USER/COCO/bu_data/trainval
cd /data/scratch/$USER/COCO/bu_data/trainval
cp /data/PublicDataSets/Coco-2014/pretrained/trainval/* .
```


Then run the `make_bu_data.py` script to extract the `npy`/`npz` files. **If running on the QMUL HPC this task takes too long to run on the login node without being killed. Therefore, amend and run the `make_bu_data.qsub` job script.**

```bash
python3 script/make_bu_data.py --output_dir /data/scratch/$USER/COCO/butd
```

Once this has run, you should have three folders in `/data/scratch/$USER/COCO/`
1. `butd_att`: Which contains an $N\times2048$-dimension numpy array of ResNet101 features for the $N$ objects detected in the image
2. `butd_fc`: Containing a $2048$-dimension ResNet101 feature for the whole image
3. `butd_box`: That contains the bounding box information for the detected objects

---

If you want to use graphs as representations for your images, keep reading! 
## VSUA/SGAE Data Download

From this [Google Drive Link](https://drive.google.com/drive/folders/1GvwpchUnfqUjvlpWTYbmEvhvkJTIWWRb), we want to download the following files
1. `coco_img_sg.zip`
2. `coco_pred_sg_rela.npy`

```bash
curl -L "https://drive.usercontent.google.com/download?id=14jk1AqnJwBXwlkfGnEgj3GyHbGcSSlij&confirm=xxx" -o coco_img_sg.zip

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1pTLfyq5nOLTqqZiKDsZ2hzlRoetfD-Y7' -O coco_pred_sg_rela.npy
```

And from this [Google Drive Link](https://drive.google.com/file/d/1G9_ZdjyIprl2wyWCExslWTWOimJf3x8G/view), we want to download
1. `vsua_box_info.pkl`

```bash
curl -L "https://drive.usercontent.google.com/download?id=1G9_ZdjyIprl2wyWCExslWTWOimJf3x8G&confirm=xxx" -o vsua_box_info.pkl
```


We then need to `unzip coco_img_sg.zip` (Once this is done it's safe to delete this zip file).

## VSUA/SGAE Data Extraction
If you are downloading on a HPC system, run `vsua.qsub`

First, run:
```bash
python3 vsua/cal_geometry_feats.py --input_file "/data/scratch/$USER/COCO/vsua_box_info.pkl" --save_path "/data/scratch/$USER/COCO/geometry_feats"
```

Finally, run the following:
```bash
python3 vsua/build_geometry_graph.py --save_path "/data/scratch/$USER/COCO/geometry-iou" --geometry_path "/data/scratch/$USER/COCO/geometry_feats"
```