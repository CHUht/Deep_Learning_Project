# Deep_Learning_Project
This repository contains the modules used for the deep learning project of CentraleSupélec

## Requirements

To execute the code, please make sure that the following packages are installed:
- [NumPy](https://docs.scipy.org/doc/numpy-1.15.1/user/install.html)
- [PyTorch and Torchvision](https://pytorch.org/) (install with CUDA if available)
- [matplotlib](https://matplotlib.org/users/installing.html)

## Executing the demos

### sparsefool_test.py
A simple demo that computes the sparse adversarial perturbation of a test image.


## Contents

### utils/
Utils function of Sparse fool method

### Grad-CAM/
Jupiter notebook of Grad-CAM analysis and some demo data


### data/
Contains some examples for the demos, including some picures from Animals10 dataset

### Model_trained
Contains trained model which serves as target for sparse fool method

### generating_noisy_image.py
Module used for generating images with the sparse adversarial perturbation

### training_using_transfer_learning.py
Module used to train target model for sparse fool, For the training we use the fine tuning of transfer learning.

### transferability_evaluation.py
Module used to evaluate the fooling rate of the sparse adversarial perturbation, this module also test the transferability between models


## Reference
[1] A. Modas, S. Moosavi-Dezfooli, P. Frossard:
*SparseFool: a few pixels make a big difference*. In Computer Vision and Pattern Recognition (CVPR ’19), IEEE, 2019.

[2] Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra:
*Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. 	arXiv:1610.02391 [cs.CV]
