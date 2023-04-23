# MH-DETR: Video Moment and Highlight Detection with Cross-modal Transformer

This is our implementation for the paper: **MH-DETR: Video Moment and Highlight Detection with Cross-modal Transformer**

![Alt text](utils/img/modal.png)

# Table of Contents

1. [Preparation](#preparation)
2. [Datasets](#datasets)
2. [Usage](#usage)
3. [Citation](#citation)

## Preparation

The released code consists of the following files.

```
MH-DETR
├── data
│   ├── activitynet
│   │   └── {train,val}.pkl
│   ├── charades
│   │   └── {train,val}.pkl
│   ├── tvsum
│   │   └── tvsum_{train,val}.jsonl
│   └── highlight_{train,val,test}_release.jsonl
├── features
│   ├── activitynet
│   │   └── c3d.hdf5
│   ├── charades
│   |   ├── vgg.hdf5
│   │   └── i3d.hdf5
|   ├──	clip_features
|   ├── clip_text_features
|   ├── slowfast_features
|   ├── tvsum
|   |   └── ...
├── mh_detr
├── standalone_eval
├── utils
├── results
├── README.md
└── ···
```

## Datasets

TODO

## Usage

### Train on QVHighlights

```sh
bash mh_detr/scripts/train.sh
```

### Inference on QVHighlights

```sh
bash mh_detr/scripts/inference.sh ${Your_Path}/MH-DETR/results/qvhighlights/model_best.ckpt val
```

Checkpoint download [link](https://drive.google.com/file/d/15Hq5zNoe51eX1M8vA_tEWWhaDlGsgoCe/view?usp=sharing). Please replace ${Your_Path} with your path. The result is as follows:

| MR R1@0.5 | MR R1@0.7 | MR mAP Avg. | HD ($\geq$ VG) mAP | HD ($\geq$ VG) HIT@1 | Params | GFLOPs |
| :-------: | :-------: | :---------: | :----------------: | :------------------: | :----: | :----: |
|   60.84   |   44.90   |    39.26    |       38.77        |        61.74         |  8.2M  |  0.34  |

### Train on other datasets

```sh
bash mh_detr/scripts/train_charades.sh --dset_name ${Dataset_Name}
```

Please replace ${Your_Path} with {activitynet, charade, tvsum}.

### Debug

```sh
bash mh_detr/scripts/train.sh --debug
```

## Citation

TDOD
