# Syntax-Aware Network for Handwritten Mathematical Expression Recognition

This is the official pytorch implementation of [SAN](https://arxiv.org/abs/2203.01601) (CVPR'2022).
![SAN Overview](overview.png)


### Environment

```
python==3.8.5
numpy==1.22.2
opencv-python==4.5.5.62
PyYAML==6.0
tensorboardX==2.5
torch==1.6.0+cu101
torchvision==0.7.0+cu101
tqdm==4.64.0
```

### Train

```
python train.py --config path_to_config_yaml
```

### Inference
```
python inference.py --config path_to_config_yaml --image_path path_to_image_folder --label_path path_to_label_folder
```

```
Example:
python inference.py --config 14.yaml --image_path data/14_test_images --label_path data/test_caption.txt
```

### Dataset

CROHME: 
```
Download the dataset from: https://github.com/JianshuZhang/WAP/tree/master/data
```

HME100K
```
Download the dataset from the official website: https://ai.100tal.com/dataset
```

### Citation

If you find this dataset helpful for your research, please cite the following paper:

```
@inproceedings{yuan2022syntax,
  title={Syntax-Aware Network for Handwritten Mathematical Expression Recognition},
  author={Yuan, Ye and Liu, Xiao and Dikubab, Wondimu and Liu, Hui and Ji, Zhilong and Wu, Zhongqin and Bai, Xiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4553--4562},
  year={2022}
}
```