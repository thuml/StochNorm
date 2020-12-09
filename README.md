# StochNorm
Original implementation for NeurIPS 2020 paper [Stochastic Normalization](https://proceedings.neurips.cc//paper/2020/file/bc573864331a9e42e4511de6f678aa83-Paper.pdf).

<p align="center">
  <img src="https://github.com/thuml/StochNorm/blob/main/arch.png" width="300">
  <br/>
  <br/>
  <a href="https://github.com/thuml/StochNorm/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/thuml/StochNorm?" /></a>
</p>

## Dependencies
* python3
* torch == 1.1.0 (with suitable CUDA and CuDNN version)
* torchvision == 0.3.0
* numpy
* argparse
* tqdm

## Datasets
| Dataset | Download Link |
| -- | -- |
| CUB-200-2011 | http://www.vision.caltech.edu/visipedia/CUB-200-2011.html |
| Stanford Cars | http://ai.stanford.edu/~jkrause/cars/car_dataset.html |
| FGVC Aircraft | http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/ |
| NIH Chest X-ray | https://nihcc.app.box.com/v/ChestXray-NIHCC |

## Quick Start
```
python --gpu [gpu_num] --data_path /path/to/dataset --class_num [class_num] --p 0.5 train.py 
```

## Some Results
We re-trained our StochNorm with this code on full 15% train data of CUB-200-2011. Results are shown in the table below.

| Sampling Rate | Top-1 Acc(%) |
| -- | -- |
| 15% | 50.41 |
| 30% | 62.09 |
| 50% | 72.05 |
| 100% | 79.65 |

## Citation
If you use this code for your research, please consider citing:
```
@article{kou2020stochastic,
  title={Stochastic Normalization},
  author={Kou, Zhi and You, Kaichao and Long, Mingsheng and Wang, Jianmin},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

## Contact
If you have any problem about our code, feel free to contact kz19@mails.tsinghua.edu.com.
