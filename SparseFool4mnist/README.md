# Sparsefool for mnist
This directory contains the code for testing sparsefool on the MNIST dataset

## Content

### Model_trained
Contains the 3 models used for testing and their accuracy graphs by epoch

### output_2(3/4)layers
Contain the output files of the sparsefool tests done on the 3 models

### Model.py
Contain the code for the model(s) used (changing number of layers)

### deepfool.py, linear_solver.py, utils.py, sparsefool.py
Contain utility code for the sparsefool_test.py

### trainModelandSave.py
Contains code for training the model on MNIST

### Sparsefool_test.py
Contains code to test sparsefool on the model trained (more explanation in the report)

### trainsferabilityTest.py
Contains code to test transferability (more in the report)

## Instructions
Use the jupyter checkpoint (.ipynb) for instructions