# SADN: Learned Light Field Image Compression with Spatial-Angular Decorrelation + QVRF: A Quantization-error-aware variable rate framework for learned image compression 

Pytorch implementation of the paper "SADN: Learned Light Field Image Compression with Spatial-Angular Decorrelation" with variable rate techniche of "QVRF: A Quantization-error-aware variable rate framework for learned image compression"

## Related links
 * CompressAI: https://github.com/InterDigitalInc/CompressAI
 * SADN：https://github.com/VincentChandelier/SADN
 * QVRF：https://github.com/bytedance/QRAF
 
## Installation

Install [CompressAI](https://github.com/InterDigitalInc/CompressAI) and the packages required for development.
```bash
conda create -n SADN+QVRF python=3.9
conda activate FPIcompress
pip install compressai==1.1.5
pip install ptflops
pip install einops
pip install tensorboardX
```

## Test
[Full-resolution test images](https://pan.baidu.com/s/14LdMV7ybwEiSauR4DlfiQA?pwd=gf66)
```bash
python3 Inference.py --dataset ./dataset/Fulltest --s 2 --output_path SADNSTE -p ./Proposed_STE__28_checkpoint.pth.tar --patch 832 --factormode 0 --factor 0.1
```

If you have any problem, please contact me: tkd20@mails.tsinghua.edu.cn

If you think it is useful for your reseach, please cite our paper.
## Citation
```
@inproceedings{tong2022sadn,
  title={SADN: learned light field image compression with spatial-angular decorrelation},
  author={Tong, Kedeng and Jin, Xin and Wang, Chen and Jiang, Fan},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1870--1874},
  year={2022},
  organization={IEEE}
}

@article{tong2023qvrf,
  title={QVRF: A Quantization-error-aware Variable Rate Framework for Learned Image Compression},
  author={Tong, Kedeng and Wu, Yaojun and Li, Yue and Zhang, Kai and Zhang, Li and Jin, Xin},
  journal={arXiv preprint arXiv:2303.05744},
  year={2023}
}
```
