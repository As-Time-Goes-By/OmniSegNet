# Omni-Referring Image Segmentation

[![arXiv](https://img.shields.io/badge/Arxiv-2512.06862-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2512.06862) [![Hugging Face Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/TUZKI/OmniRef)

## 🗓️TODO

- [x] Release training codes and OmniRef dataset.
- [x] Release paper.

## 🛠️ Installation

- Create a conda virtual environment and activate it
```bash
conda create -n omnisegnet python=3.8 -y
conda activate omnisegnet
```

- Install Pytorch following the [official installation instructions](https://pytorch.org/get-started/previous-versions)

```bash
# CUDA 11.3
pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

- Install Detectron following the [official installation instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
  
```bash
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
```

- Compile the MSDeformAttn layer:
  
```bash
cd OmniSegNet_model/modeling/pixel_decoder/ops
sh make.sh
```

```bash
pip install -r requirements.txt
```

```bash
wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
pip install en_vectors_web_lg-2.1.0.tar.gz
pip install albumentations
pip install Pillow==9.5.0
pip install tensorboardX
```

## 📚Data Preparation

- The data structure should look like the following:

```
| -- datasets
        | -- anns
            | -- omniRef_train/val.json
            | -- grefs(unc).json
        | -- images
            | -- train2014
                | -- COCO_train2014_XXXXX.jpg
                | -- ...
  
```
## 🚀Training
Firstly, download the backbone weights (`swin_base_patch4_window12_384_22k.pth`) and  (`bert-base-uncased`).
```
sh train.sh
```

## 🤝 Acknowledgments

This project is based on [refer](https://github.com/lichengunc/refer), [ReLA](https://github.com/henghuiding/ReLA), [Detectron2](https://github.com/facebookresearch/detectron2), [VRP-SAM](https://github.com/syp2ysy/VRP-SAM). Many thanks to the authors for their great works!


## ✏️ Citation

If you find our paper and code helpful, we kindly invite you to give it a star and consider citing our work.

```bibtex
@article{zheng2025omni,
  title={Omni-Referring Image Segmentation},
  author={Zheng, Qiancheng and Shen, Yunhang and Luo, Gen and Song, Baiyang and Sun, Xing and Sun, Xiaoshuai and Zhou, Yiyi and Ji, Rongrong},
  journal={arXiv preprint arXiv:2512.06862},
  year={2025}
}
```
