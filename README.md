# MVMS-RCN: A Dual-Domain Unified CT Reconstruction with Multi-sparse-view and Multi-scale Refinement-correction

This repository contains sparse-view CT reconstruction pytorch codes for the following paper：  
Xiaohong Fan, Ke Chen, Huaming Yi, Yin Yang*, and Jianping Zhang*, “MVMS-RCN: A Dual-Domain Unified CT Reconstruction with Multi-sparse-view and Multi-scale Refinement-correction”, IEEE Transactions on Computational Imaging, 2024,10:1749-1762. DOI: 10.1109/TCI.2024.3507645. [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10769006) 

Xiaohong Fan, Ke Chen, Huaming Yi, Yin Yang*, and Jianping Zhang*, “MVMS-RCN: A Dual-Domain Unified CT Reconstruction with Multi-sparse-view and Multi-scale Refinement-correction”, arXiv, May 2024. [[pdf]](https://arxiv.org/pdf/2405.17141) 

These codes are built on PyTorch and tested on Ubuntu 18.04/20.04 (Python3.x, PyTorch>=0.4) with Intel Xeon Silver 4214 CPU and Tesla V100-PCIE-32GB GPU.

### Abstract
X-ray Computed Tomography (CT) is one of the most important diagnostic imaging techniques in clinical applications. Sparse-view CT imaging reduces the number of projection views to a lower radiation dose and alleviates the potential risk of radiation exposure. Most existing deep learning (DL) and deep unfolding sparse-view CT reconstruction methods: 1) do not fully use the projection data; 2) do not always link their architecture designs to a mathematical theory; 3) do not flexibly deal with multi-sparse-view reconstruction assignments. This paper aims to use mathematical ideas and design optimal DL imaging algorithms for sparse-view CT reconstructions. We propose a novel dual-domain unified framework that offers a great deal of flexibility for multi-sparse-view CT reconstruction through a single model. This framework combines the theoretical advantages of model-based methods with the superior reconstruction performance of DL-based methods, resulting in the expected generalizability of DL. We propose a refinement module that utilizes unfolding projection domain to refine full-sparse-view projection errors, as well as an image domain correction module that distills multi-scale geometric error corrections to reconstruct sparse-view CT. This provides us with a new way to explore the potential of projection information and a new perspective on designing network architectures. The multi-scale geometric correction module is end-to-end learnable, and our method could function as a plug-and-play reconstruction technique, adaptable to various applications. Extensive experiments demonstrate that our framework is superior to other existing state-of-the-art methods.

![Fig1 ChartFlow_00](https://github.com/fanxiaohong/MVMS-RCN/blob/main/ChartFlow.pdf)
Fig. 1. The overall architecture of the proposed unified dual-domain multisparse-view CT reconstruction framework (MVMS-RCN). It consists of multiview projection refinement module R and multi-scale geometric correction module D.

### Environment  
```
pytorch = 1.7.1 (recommend 1.6.0, 1.7.1)
scikit-image <= 0.16.2 (recommend 0.16.1, 0.16.2)
torch-radon = 1.0.0 (for sparse-view CT)
```
### Prepare data
Due to upload file size limitation, we are unable to upload data directly. Here we provide a [link](https://pan.baidu.com/s/1baOAEXmHZdsulsCKKgNbsg?pwd=io4f) to download the datasets for you. 

### 1.Test sparse-view CT  
The torch-radon package (pip install torch-radon) is necessary for sparse-view CT reconstruction.    
3.1、Pre-trained models:  
All pre-trained models for our paper are in './model and result fan CT HU'.  
3.2、Prepare test data:  
Due to upload file size limitation, we are unable to upload data directly. Here we provide a [link](https://pan.baidu.com/s/1baOAEXmHZdsulsCKKgNbsg?pwd=io4f) to download the datasets for you.   
3.3、Prepare code:  
Open './Core_CT-proposed-MVMS_RCN.py' and change the default run_mode to test in parser (parser.add_argument('--run_mode', type=str, default='test', help='train or test')).  
3.4、Run the test script (Core_CT-proposed-MVMS_RCN.py).  
3.5、Check the results in './result/'.

### 2.Train sparse-view CT   
4.1、Prepare training data:  
Due to upload file size limitation, we are unable to upload training data directly. Here we provide a [link](https://pan.baidu.com/s/1baOAEXmHZdsulsCKKgNbsg?pwd=io4f) to download the datasets for you.  
4.2、Prepare code:  
Open './Core_CT-proposed-MVMS_RCN.py' and change the default run_mode to train in parser (parser.add_argument('--run_mode', type=str, default='train', help='train or test')).  
4.3、Run the train script (Core_CT-proposed-MVMS_RCN.py).  
4.4、Check the results in './log_CT/'.

### Citation  
If you find the code helpful in your resarch or work, please cite the following papers. 
```
@Article{Fan2024,
  author  = {Xiaohong Fan and Ke Chen and Huaming Yi and Yin Yang and Jianping Zhang},
  journal = {IEEE Transactions on Computational Imaging},
  title   = {MVMS-RCN: A Dual-Domain Unified CT Reconstruction with Multi-sparse-view and Multi-scale Refinement-correction},
  year    = {2024},
  month   = {Nov.},
  pages   = {1749--1762},
  volume  = {10},
  doi     = {10.1109/TCI.2024.3507645},
}
```

### Acknowledgements  
Thanks to the authors of ISTA-Net and FISTA-Net, our codes are adapted from the open source codes of them.   

### Contact  
The code is provided to support reproducible research. If the code is giving syntax error in your particular python configuration or some files are missing then you may open an issue or directly email me at fanxiaohong@zjnu.edu.cn or fanxiaohong1992@gmail.com or fanxiaohong@smail.xtu.edu.cn
