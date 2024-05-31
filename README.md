# pcs-stereo

This repository is code for our paper "Proxy-supervised Cross-spectral Stereo Matching"

The pipeline of our method is as follows:

![pipeline](https://github.com/jiayuzhang128/pcs-stereo/blob/master/imgs/overall.png)

## Enviroment

Our experiments were conducted in the following environments:

+ Nvidia GForce 3090 * 1

+ Ubuntu 18.04

+ Python 3.8

+ Pytorch

For detailed environment configuration, please refer to `requirements.txt`

### special dependency installation

```bash
pip install 'git+https://github.com/saadnaeem-dev/pytorch-linear-warmup-cosine-annealing-warm-restarts-weight-decay'
```

## Data preparation

We use `Pittsburgh` cross-spectrial stereo dataset, please refer to [DMC](https://github.com/tiancheng-zhi/cs-stereo) for downloading.

To generate **dense pseudo-labels**， please first follow the steps in the paper to generate initial labels using [Metric3D](https://github.com/YvanYin/Metric3D) and [CREStereo](https://github.com/ibaiGorordo/CREStereo-Pytorch), then refer to our code in `pseudo_label_generation` folder.

more qualitative results of our **pseudo-label generation** method are shown in `imgs/results.pdf`.

## Pretrained Model

Download [pretrained models](https://drive.google.com/drive/folders/1_iGUJqVaIl5yZdRfXC3cZCBxiqDFGzig?usp=drive_link)

Performance (RMSE, lower is better):

| Model | Common | Light | Glass | Glossy | Vegetation | Skin | Clothing | Bag | Mean |
|:-------|:-----:|:-----:|:-----:|:------:|:----------:|:-----:|:-------:|:---:|:----:|
| PSMNet* | 0.45 | 0.79 | 0.83 | 0.99 | 0.65 | 0.83 | 0.83 | 0.59 | 0.74 |
| IGEVStereo* |0.42 | 0.46 | 0.82 | 0.95 | 0.59 | 0.58 | 0.44 | 0.50 | 0.60|

> '\*' denotes the conventional stereo-matching network trained using our method.

## Train and evaluation

Please refer to `run.ipynb` for details.