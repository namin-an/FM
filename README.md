# Feedback Mechanism (FM) in Machine Learning to Reduce Cost of Human Psychophysical Test of Prosthetic Vision
Na Min An, Hyeonhee Roh, Sein Kim, Jae Hun Kim, and Maesoon Im

## Main figure
<p align="center" width="100%"><img src="https://github.com/namin-an/FM/blob/main/images/Fig1.png"></img></p>   
🌃: https://pixabay.com
🌁: https://www.flaticon.com
<br />

## Abstract video
[![IMAGE ALT TEXT](https://github.com/namin-an/FM/blob/main/images/cover.png)](https://www.youtube.com/watch?v=kHdlyUNurds)
<br />

## Dependencies
PyTorch (1.9.0), NumPy (1.20.3), matplotlib (3.4.3), pandas (1.3.2), sklearn (0.23.2), skimage (0.17.2).
<br />

## Dataset
We used K-face dataset, which can be downloaded from https://aihub.or.kr.
<br />

## Code structure

### Train and test models
```
├── test.py  
│   ├── loadData.py  
│   │   ├── loadData  
│   │   ├── visualizeData  
│   ├── trainANNs.py  
│   │   ├── beginModeling  
```

### Visualize results
```
├── visualizing_fig2.py  
├── visualizing_fig3.py  
├── visualizing_fig4.py  
├── fm.py  
```

### Adaptations
> Grad-CAM [*/pytorch_grad_cam*](https://github.com/jacobgil/pytorch-grad-cam)   
> Early-stopping [*/mypackages/pytorchtools.py*](https://github.com/Bjarten/early-stopping-pytorch)
<br />

If you have any questions or comments regarding the code, please do not hesitate to contact me through this email: **namin0202@gmail.com**
