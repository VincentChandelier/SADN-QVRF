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

## Trainning
### stage 1
Train a high-rate fixed-rate model. 
```
python train.py -d dataset --N 48 --angRes 13 --n_blocks 1 -e 100 -lr 1e-4 -n 20  --lambda 3e-3 --batch-size 8  --test-batch-size 8 --aux-learning-rate 1e-3 --patch-size 832 832 --cuda --save --seed 1926 --clip_max_norm 1.0 --gpu-id  2,3 --savepath   ./checkpoint --stage 1 --ste 0 --loadFromPretrainedSinglemodel 0  
```

### stage 2
Load the last checkpoint of stage 1 or load from the [Fixed-rate model of SADN](https://drive.google.com/file/d/1Pb-uJw497ho7XxBsWqsOG4c1cwAKXdDc/view?usp=sharing) and begin to train the variabal rate model in stage 2.  
```
python train.py -d dataset --N 48 --angRes 13 --n_blocks 1 -e 100 -lr 1e-4 -n 20  --lambda 3e-3 --batch-size 8  --test-batch-size 8 --aux-learning-rate 1e-3 --patch-size 832 832 --cuda --save --seed 1926 --clip_max_norm 1.0 --gpu-id  2,3 --savepath  ./Noisecheckpoint --stage 2 --ste 0 --loadFromPretrainedSinglemodel 1 --checkpoint checkpoint.pth.tar --pretrained
```
### stage 3  
Load the last checkpoint of stage 2 and begin to train the final variabal rate model in stage 3.
```
python train.py -d dataset --N 48 --angRes 13 --n_blocks 1 -e 100 -lr 1e-6 -n 20  --lambda 3e-3 --batch-size 8  --test-batch-size 32 --aux-learning-rate 1e-3 --patch-size 832 832 --cuda --save --seed 1926 --clip_max_norm 1.0 --gpu-id  2,3 --savepath  ./STEcheckpoint --stage 3 --ste 1 --loadFromPretrainedSinglemodel 0 --checkpoint ./Noisecheckpoint/checkpoint_best_loss_82.pth.tar --pretrained
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
