# MVMS-RCN: A Dual-Domain Unfolding CT Reconstruction with Multi-sparse-view and Multi-scale Refinement-correction

This repository contains sparse-view CT reconstruction pytorch codes for the following paper：  
Xiaohong Fan, Ke Chen, Huaming Yi, Yin Yang*, and Jianping Zhang*, “MVMS-RCN: A Dual-Domain Unfolding CT Reconstruction with Multi-sparse-view and Multi-scale Refinement-correction”, IEEE Transactions on Computational Imaging, 2024. Accept.

Xiaohong Fan, Ke Chen, Huaming Yi, Yin Yang*, and Jianping Zhang*, “MVMS-RCN: A Dual-Domain Unfolding CT Reconstruction with Multi-sparse-view and Multi-scale Refinement-correction”, arXiv, May 2024. [[pdf]](https://arxiv.org/pdf/2405.17141) 

These codes are built on PyTorch and tested on Ubuntu 18.04/20.04 (Python3.x, PyTorch>=0.4) with Intel Xeon Silver 4214 CPU and Tesla V100-PCIE-32GB GPU.

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
@Article{Fan2023,
  author  = {Xiaohong Fan and Ke Chen and Huaming Yi and Yin Yang and Jianping Zhang},
  journal = {},
  title   = {MVMS-RCN: A Dual-Domain Unfolding CT Reconstruction with Multi-sparse-view and Multi-scale Refinement-correction},
  year    = {2024},
  month   = {},
  pages   = {},
  volume  = {},
  doi     = {},
}
```

### Acknowledgements  
Thanks to the authors of ISTA-Net and FISTA-Net, our codes are adapted from the open source codes of them.   

### Contact  
The code is provided to support reproducible research. If the code is giving syntax error in your particular python configuration or some files are missing then you may open an issue or directly email me at fanxiaohong@zjnu.edu.cn or fanxiaohong1992@gmail.com or fanxiaohong@smail.xtu.edu.cn
