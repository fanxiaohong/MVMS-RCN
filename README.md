# MVMS-RCN: A Deep Unfolding CT Reconstruction with Multi-sparse-view and Multi-scale Refinement-correction

This repository contains sparse-view CT reconstruction pytorch codes for the following paper：  
Xiaohong Fan, Yin Yang, Ke Chen, Huaming Yi, and Jianping Zhang*, “MVMS-RCN: A Deep Unfolding CT Reconstruction with Multi-sparse-view and Multi-scale Refinement-correction”, 2023.  Under review.

These codes are built on PyTorch and tested on Ubuntu 18.04/20.04 (Python3.x, PyTorch>=0.4) with Intel Xeon CPU E5-2630 and Nvidia Tesla V100 GPU.

### Environment  
```
pytorch <= 1.7.1 (recommend 1.6.0, 1.7.1)
scikit-image <= 0.16.2 (recommend 0.16.1, 0.16.2)
torch-radon = 1.0.0 (for sparse-view CT)
```
