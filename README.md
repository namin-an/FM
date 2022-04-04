# Feedback Mechanism (FM) in Machine Learning to Reduce Cost of Human Psychophysical Test of Prosthetic Vision
Na Min An, Hyeonhee Roh, Sein Kim, Jae Hun Kim, and Maesoon Im

## Dependencies
PyTorch (1.9.0), NumPy (1.20.3), matplotlib (3.4.3), pandas (1.3.2), sklearn (0.23.2), skimage (0.17.2).

## Dataset
We used K-face dataset, which can be downloaded from https://aihub.or.kr.

## Codes
### Loading data:
* loadData.py

### Training models:
* trainANNs.py

### Implementation of FM:
* fm.py

### Plotting figures:
* visualizing_fig2.py
* visualizing_fig3.py
* visualizing_fig4.py

### Adaptations:
* Grad-CAM for XAI visualizations (*/pytorch_grad_cam*) from https://github.com/jacobgil/pytorch-grad-cam. <br />
* Early-stopping (*/mypackages/pytorchtools.py*) from https://github.com/Bjarten/early-stopping-pytorch.

## Example run
```
python3 test.py
```
