# MFMGC: A Multi-Modal Data Fusion Model for Movie Genre Classification

This repository contains the source code of MFMGC and the way of obtaining the dataset MovieBricks.

## Requirements

The detail can be found in the file **requirements.txt**.

```bash
conda create -n mgc -c anaconda python=3.7.2
conda activate mgc
pip install -r requirements.txt
```

## Data preparation

The complete dataset will be released when the paper is published.

Note:

**Before training, you should download [swin_small](https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth) to the folder ``./models/cache/swin/''.**

Also, change the models/MFMGC/model.py "line-13" (sys.path.append(...)) to the **absolute path** of ./models in your machine.



## Training

```python
python train.py --model_name mymodel --modals summary
```

## Results

