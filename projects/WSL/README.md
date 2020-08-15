# Enabling Deep Residual Networks for Weakly Supervised Object Detection

By [Yunhang Shen](), [Rongrong Ji](), [Yan Wang](), [Zhiwei Chen](), [Feng Zheng](), [Feiyue Huang](), [Yunsheng Wu]().

ECCV 2020 Paper.

This project is based on [Detectron2](https://github.com/facebookresearch/detectron2).

## License

DRN-WSOD is released under the [Apache 2.0 license](LICENSE).


## Citing DRN-WSOD

If you find DRN-WSOD useful in your research, please consider citing:

```
@inproceedings{DRN-WSOD_2020_ECCV,
	author = {Shen, Yunhang and Ji, Rongrong and Wang, Yan and Chen, Zhiwei and Zheng, Feng and Huang, Feiyue and Wu, Yunsheng},
	title = {Enabling Deep Residual Networks for Weakly Supervised Object Detection},
	booktitle = {European Conference on Computer Vision (ECCV)},
	year = {2020},
}   
```

## Installation

Install our forked Detectron2:
```
git clone https://github.com/shenyunhang/DRN-WSOD-pytorch.git
cd DRN-WSOD-pytorch
python3 -m pip install -e .
```
If you have problem of installing Detectron2, please checking [the instructions](https://detectron2.readthedocs.io/tutorials/install.html).

Install DRN-WSOD project:
```
cd projects/WSL
pip3 install -r requirements.txt
git submodule update --init --recursive
python3 -m pip install -e .
cd ../../
```

## Dataset Preparation
Please follow [this](https://github.com/shenyunhang/DRN-WSOD-pytorch/blob/DRN-WSOD/datasets/README.md#expected-dataset-structure-for-pascal-voc) to creating symlinks for PASCAL VOC.

Download MCG proposal from [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/) to detectron/datasets/data, and transform it to pickle serialization format:

```
cd datasets/proposals
tar xvzf MCG-Pascal-Main_trainvaltest_2007-boxes.tgz
cd ../../
python3 projects/WSL/tools/proposal_convert.py voc_2007_train datasets/proposals/MCG-Pascal-Main_trainvaltest_2007-boxes datasets/proposals/mcg_voc_2007_train_d2.pkl
python3 projects/WSL/tools/proposal_convert.py voc_2007_val datasets/proposals/MCG-Pascal-Main_trainvaltest_2007-boxes datasets/proposals/mcg_voc_2007_val_d2.pkl
python3 projects/WSL/tools/proposal_convert.py voc_2007_test datasets/proposals/MCG-Pascal-Main_trainvaltest_2007-boxes datasets/proposals/mcg_voc_2007_test_d2.pkl
```


## Model Preparation

Download models from this [here](https://1drv.ms/f/s!Am1oWgo9554dgRQ8RE1SRGvK7HW2):
```
mv models $DRN-WSOD
```

Then we have the following directory structure:
```
DRN-WSOD
|_ models
|  |_ DRN-WSOD
|     |_ resnet18_ws_model_120.pkl
|     |_ resnet150_ws_model_120.pkl
|     |_ resnet101_ws_model_120.pkl
|_ ...
```


## Quick Start: Using DRN-WSOD

### WSDDN

#### ResNet18-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/wsddn_WSR_18_DC5_1x.yaml OUTPUT_DIR output/wsddn_WSR_18_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet50-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/wsddn_WSR_50_DC5_1x.yaml OUTPUT_DIR output/wsddn_WSR_50_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet101-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/wsddn_WSR_101_DC5_1x.yaml OUTPUT_DIR output/wsddn_WSR_101_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### VGG16
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/wsddn_V_16_DC5_1x.yaml OUTPUT_DIR output/wsddn_V_16_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

### OICR

#### ResNet18-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/oicr_WSR_18_DC5_1x.yaml OUTPUT_DIR output/oicr_WSR_18_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet50-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/oicr_WSR_50_DC5_1x.yaml OUTPUT_DIR output/oicr_WSR_50_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet101-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/oicr_WSR_101_DC5_1x.yaml OUTPUT_DIR output/oicr_WSR_101_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### VGG16
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/oicr_V_16_DC5_1x.yaml OUTPUT_DIR output/oicr_V_16_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

### OICR + Box Regression

#### ResNet18-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/reg/oicr_WSR_18_DC5_1x.yaml OUTPUT_DIR output/oicr_reg_WSR_18_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet50-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/reg/oicr_WSR_50_DC5_1x.yaml OUTPUT_DIR output/oicr_reg_WSR_50_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet101-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/reg/oicr_WSR_101_DC5_1x.yaml OUTPUT_DIR output/oicr_reg_WSR_101_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### VGG16
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/reg/oicr_V_16_DC5_1x.yaml OUTPUT_DIR output/oicr_reg_V_16_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

### PCL

#### ResNet18-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/pcl_WSR_18_DC5_1x.yaml OUTPUT_DIR output/pcl_WSR_18_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet50-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/pcl_WSR_50_DC5_1x.yaml OUTPUT_DIR output/pcl_WSR_50_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet101-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/pcl_WSR_101_DC5_1x.yaml OUTPUT_DIR output/pcl_WSR_101_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### VGG16
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/pcl_V_16_DC5_1x.yaml OUTPUT_DIR output/pcl_V_16_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```


