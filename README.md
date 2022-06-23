# 3DLG-Detector
3D Object Detection via Simultaneous Local-Global Feature Learning. This work is developed on the top of [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) toolbox and includes the models and results on SUN RGB-D and ScanNet datasets in the paper.

## Installation and Usage

### Dependencies
- NVIDIA GPU + CUDA 11.1
- Python 3.8 (Recommend to use Anaconda)
- PyTorch == 1.8.0
- [mmcv-full](https://github.com/open-mmlab/mmcv) == 1.4.0
- [mmdet](https://github.com/open-mmlab/mmdetection) == 2.19.0
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) == 0.20.0

### Installation
1. Install dependencies following their guidelines.
2. Clone and install [mmdet3d](https://github.com/open-mmlab/mmdetection3d) in develop mode.

```
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
python setup.py develop
```

3. Add the files in this repo into the directories in mmdet3d.

### Training and Testing

Download the pretrained weights from [Baidu]() and put them in the `checkpoints` folder. Use `votenet_3dlg_sunrgbd-3d-10class` as an example:

```
# Training
bash -x tools/dist_train.sh configs/3dlg/votenet_3dlg_sunrgbd-3d-10class.py 8

# Testing 
bash tools/dist_test.sh configs/3dlg/votenet_3dlg_sunrgbd-3d-10class.py checkpoints/votenet_ptr_sunrgbd-3d-10class.pth 8 --eval mAP
```

