Prepare the files in the following structure.

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
│   ├── charades
│   │   └── id3.hdf5
|	└──	TODO...
├── mh_detr
├── standalone_eval
├── utils
├── results
├── README.md
└── ···
```

**Train on QVHighlights**

```sh
bash mh_detr/scripts/train.sh
```

Debug

```sh
bash mh_detr/scripts/train.sh --debug
```

**Inference on QVHighlights**

```sh
bash mh_detr/scripts/inference.sh ${Your_Path}/MH-DETR/results/qvhighlights/model_best.ckpt val
```

Checkpoint download [link](https://drive.google.com/file/d/15Hq5zNoe51eX1M8vA_tEWWhaDlGsgoCe/view?usp=sharing). Please replace ${Your_Path} with your path. The result is as follows:

| MR R1@0.5 | MR R1@0.7 | MR mAP Avg. | HD ($\geq$ VG) mAP | HD ($\geq$ VG) HIT@1 | Params | GFLOPs |
| :-------: | :-------: | :---------: | :----------------: | :------------------: | :----: | :----: |
|   60.84   |   44.90   |    39.26    |       38.77        |        61.74         |  8.2M  |  0.34  |

