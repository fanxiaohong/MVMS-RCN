# MVMS-RCN: A Deep Unfolding CT Reconstruction with Multi-sparse-view and Multi-scale Refinement-correction

This repository contains sparse-view CT reconstruction pytorch codes for the following paper：  
Xiaohong Fan, Yin Yang, Ke Chen, Huaming Yi, and Jianping Zhang*, “MVMS-RCN: A Deep Unfolding CT Reconstruction with Multi-sparse-view and Multi-scale Refinement-correction”, 2023.  Under review.

These codes are built on PyTorch and tested on Ubuntu 18.04/20.04 (Python3.x, PyTorch>=0.4) with Intel Xeon CPU E5-2630 and NVIDIA GeForce GTX 1080Ti GPU.

### Environment  
```
pytorch <= 1.7.1 (recommend 1.6.0, 1.7.1)
scikit-image <= 0.16.2 (recommend 0.16.1, 0.16.2)
torch-radon = 1.0.0 (for sparse-view CT)
```



### Citation  
If you find the code helpful in your resarch or work, please cite the following papers. 
```
@Article{Fan2023,
  author  = {Xiaohong Fan and Yin Yang and Ke Chen and Huaming Yi and Jianping Zhang},
  journal = {},
  title   = {MVMS-RCN: A Deep Unfolding CT Reconstruction with Multi-sparse-view and Multi-scale Refinement-correction},
  year    = {2023},
  month   = {},
  pages   = {},
  volume  = {},
  doi     = {},
}
```

### Acknowledgements  
Thanks to the authors of FISTA-Net, our codes are adapted from the open source codes of them.   

### Contact  
The code is provided to support reproducible research. If the code is giving syntax error in your particular python configuration or some files are missing then you may open an issue or directly email me at fanxiaohong@smail.xtu.edu.cn
