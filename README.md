# MWRN

Kan Chang, Minghong Li, Pak Lun Kevin Ding, Baoxin Li, "**Accurate Single Image Super-Resolution Using Multi-Path Wide-Activated Residual Network**", Signal Processing 2020
[[Paper]](https://www.sciencedirect.com/science/article/abs/pii/S0165168420301109)

The code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and tested on Ubuntu 18.04 environment (Python3.6, PyTorch_1.0.1, CUDA10.0, cuDNN7.4) with 1080Ti GPUs. 

## Dependencies
* Python 3.6
* Pytorch 1.0.1
* numpy
* skimage
* imageio
* matplotlib
* tqdm

If you want to use our conda environment, you can run the following scripts to create the conda environment.
  ```
  Cd to './MWRN-master'
  conda env create -f environment.yml
  ```

## Contents
1. [Abstract](#abstract)
2. [Architecture](#architecture)
3. [Train](#train)
4. [Test](#test)
5. [Note](#note)
6. [Results](#results)
7. [Citation](#citation)
8. [Acknowledgements](#acknowledgements)

## Abstract
In many recent image super-resolution (SR) methods based on convolutional neural networks
(CNNs), the superior performance was achieved by training very large networks, which may not
be suitable for real-world applications with limited computing resources. Therefore, it is necessary
to develop more compact networks that achieve a better trade-off between the model size and the
performance. In this paper, we propose an efficient and effective network called multi-path wideactivated residual network (MWRN). Firstly, as the basic building block of MWRN, the multi-path
wide-activated residual block (MWRB) is presented to extract the multi-scale features. MWRB
consists of three parallel wide-activated residual paths, where the dilated convolutions with different
dilation factors are used to increase the receptive fields. Secondly, the fusional channel attention
(FCA) module, which contains a bottleneck layer and a multi-path wide-activated residual channel
attention (MWRCA) block, is designed to well exploit the multi-level features in MWRN. In each
FCA, the MWRCA block refines the fused features by taking the interdependencies among feature
channels into consideration. The experiments demonstrate that, compared with the state-of-theart methods, the proposed MWRN model is able to provide very competitive performance with a
relatively small number of parameters.

![trade-off](/Figs/paramete_psnr_methods.png)
Performance comparisons among different SR models. 

## Architecture

![MWRB](/Figs/multi-scale_learning_blocks.png)
The structures of different multi-scale learning blocks.

![MWRN](/Figs/structures_MWRN_FCA.png)
The structures of the MWRN and the FCA module.

## Data

Training data [DIV2K (800 training + 100 validtion images)]

Benchmark data [(Set5, Set14, B100, Urban100, Manga109)]

## Train

1. Download DIV2K training data from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) and place them in './DATA/'.

2. Cd to './code/src', run the following scripts to train models.

    ```bash
    #-------------MWRN_L_x2 
    python main.py --model MWRN_L --scale 2  --save MWRN_L_x2  --epochs 1000 --reset --patch_size 96 --cudnn
    #-------------MWRN_L_x3 
    python main.py --model MWRN_L --scale 3  --save MWRN_L_x3  --epochs 1000 --reset --patch_size 144 --cudnn
    #-------------MWRN_L_x4 
    python main.py --model MWRN_L --scale 4  --save MWRN_L_x4  --epochs 1000 --reset --patch_size 192 --cudnn
    #-------------MWRN_M_x2
    python main.py --model MWRN_M --scale 2  --save MWRN_M_x2  --epochs 1000 --reset --patch_size 96 --cudnn
    #-------------MWRN_M_x3
    python main.py --model MWRN_M --scale 3  --save MWRN_M_x3  --epochs 1000 --reset --patch_size 144 --cudnn
    #-------------MWRN_M_x4 
    python main.py --model MWRN_M --scale 4  --save MWRN_M_x4  --epochs 1000 --reset --patch_size 192 --cudnn
    #-------------MWRN_H_x2
    python main.py --model MWRN_H --scale 2  --save MWRN_H_x2  --epochs 1000 --reset --patch_size 96 --cudnn
    #-------------MWRN_H_x3
    python main.py --model MWRN_H --scale 3  --save MWRN_H_x3  --epochs 1000 --reset --patch_size 144 --cudnn
    #-------------MWRN_H_x4
    python main.py --model MWRN_H --scale 4  --save MWRN_H_x4  --epochs 1000 --reset --patch_size 192 --cudnn

    ```

## Test

1. Download bechmark data from [Google Driver](https://drive.google.com/open?id=12tkfpxh-CJx33Z3GNOOUvNQ9KfQWlPQL) or [BaiDuYun](https://pan.baidu.com/s/1zsWWoW2WEl-d5hksKkhWJA) (password：df24). Then place them in './DATA/'.

2. Cd to './code/src', run the following scripts to test models.

    ```bash
    # No self-ensemble: MWRN
    # Scale 2,3,4
    #-------------MWRN_L_x2 
    python main.py --model MWRN_L --scale 2  --pre_train ../experiment/model/MWRN_L_x2.pt --save MWRN_L_x2 --test_only --data_test Set5
    #-------------MWRN_L_x3 
    python main.py --model MWRN_L --scale 3  --pre_train ../experiment/model/MWRN_L_x3.pt --save MWRN_L_x3 --test_only --data_test Set5
    #-------------MWRN_L_x4 
    python main.py --model MWRN_L --scale 4  --pre_train ../experiment/model/MWRN_L_x4.pt --save MWRN_L_x4 --test_only --data_test Set5
    #-------------MWRN_M_x2 
    python main.py --model MWRN_M --scale 2  --pre_train ../experiment/model/MWRN_M_x2.pt --save MWRN_M_x2 --test_only --data_test Set5
    #-------------MWRN_M_x3
    python main.py --model MWRN_M --scale 3  --pre_train ../experiment/model/MWRN_M_x3.pt --save MWRN_M_x3 --test_only --data_test Set5
    #-------------MWRN_M_x4
    python main.py --model MWRN_M --scale 4  --pre_train ../experiment/model/MWRN_M_x4.pt --save MWRN_M_x4 --test_only --data_test Set5
    #-------------MWRN_H_x2
    python main.py --model MWRN_H --scale 2  --pre_train ../experiment/model/MWRN_H_x2.pt --save MWRN_H_x2 --test_only --data_test Set5
    #-------------MWRN_H_x3
    python main.py --model MWRN_H --scale 3  --pre_train ../experiment/model/MWRN_H_x3.pt --save MWRN_H_x3 --test_only --data_test Set5
    #-------------MWRN_H_x4
    python main.py --model MWRN_H --scale 4  --pre_train ../experiment/model/MWRN_H_x4.pt --save MWRN_H_x4 --test_only --data_test Set5

    ```

4. Cd to './code/experiment'   Run 'Evaluate_PSNR_SSIM.m' to obtain PSNR/SSIM values for paper.

## Note
### Prepare beachmark data
* if you want to generate new HR/LR images, you can run 'Prepare_TestData_HR_LR.m' in Matlab
### Evaluate the results
* if you want to evaluate PSNR and SSIM for each dataset, you need to save images by appending '--save_result' to test commands.
* if you want to obtain the results of MWRN+, you need to save images by appending '--self-ensemble' to test commands.

## Results
Visual results can be downloaded from [BaiDuYun](https://pan.baidu.com/s/12BdVooik6ii3PxtC62S4Xw) (password: ux4k)

### Quantitative Results
![PSNR_SSIM_Lightweight](/Figs/psnr_lightweight.png)
Quantitative comparison among different lightweight models (PSNR (dB) /SSIM). Best and second best results are highlighted

![PSNR_SSIM_Middleweight and Heavyweight](/Figs/psnr_middleweight_heavyweight.png)
Quantitative comparison among middleweight and heavyweight models (PSNR (dB)/SSIM). Best and second best results are highlighted

### Visual Results

![Visual_Result](/Figs/image062_Urban100_x3.png)
![Visual_Result](/Figs/image073_Urban100_x3.png)
![Visual_Result](/Figs/image024_Urban100_x3.png)
![Visual_Result](/Figs/62096_BSD100_x3.png)
Visual comparisons for ×3 scale

![Visual_Result](/Figs/KoukouNoHitotachi_Manga109_x4.png)
![Visual_Result](/Figs/image020_Urban100_x4.png)
Visual comparisons for ×4 scale

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@article{Chang_2020,
  title={Accurate Single Image Super-Resolution Using Multi-Path Wide-Activated Residual Network},
  author={K. Chang, M. Li, P. L. K. Ding, B. Li},
  journal={Signal Processing},
  volume={172},
  number = {107567},
  pages={},
  year={2020},
  publisher={Elsevier}
}

```
## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). 
We also refer to some other work such as [WDSR](https://github.com/JiahuiYu/wdsr_ntire2018), [RCAN](https://github.com/yulunzhang/RCAN), [AWSRN](https://github.com/ChaofWang/AWSRN), [pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter)
We thank these authors for sharing their codes.



