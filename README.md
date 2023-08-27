## "Distribution-informed Learning for Fluorescence Microscopy Restoration"

## System Requirements

We highly recommend using the Linux operating system. Using an nVIDIA GPU with >=11GB memory is highly recommended, although this tool can be used with CPU only.

We used Ubuntu 9.4.0 (GNU/Linux 5.15.0-76-generic x86_64)) and an NVIDIA Corporation Device 2204 (rev a1) GPU and CUDA version 11.4.


## Installation
* Python 3.7
* Packages:
  
  basicsr          ==          1.4.2

  easydict         ==          1.11.dev0

  imageio          ==          2.13.3

  keras            ==          2.11.0

  numpy            ==          1.21.5

  opencv-python    ==          4.5.4.60

  Pillow           ==          9.0.1

  scikit-image     ==          0.19.2

  scipy            ==          1.7.3

  tensorflow-gpu   ==          2.7.0

  tifffile         ==          2021.11.2

  torch            ==          1.10.0+cu113
  
  csbdeep [![PyPI version](https://badge.fury.io/py/csbdeep.svg)](https://pypi.org/project/csbdeep)


#### Training the model

  The overall training process can be divided into two phases. In the first training phase, the model is optimized with an uncertainty-driven loss (UDL) to learn the uncertainty of the IR results and the common L2 norm loss to calculate the similarity between the IR results and the HR images. In the second training phase, the uncertainty-weighted (UWL) loss fine-tunes the model to focus more on image areas with larger reconstruction uncertainty. The well-trained model in the first phase is frozen to provide the uncertainty for calculating the weight in UWL loss.

- The first stage of training the model for denoising

```
cd ./denoise/
python training_TF.py
```
* Modify `training_TF.py` 
  istrain=True
  modeltypelst = ['FBuncertainty_f_uncer']
Replacing "traindatapath" with the path of the training data `data_label.npz`.

- The second stage of training the model for denoising

```
cd ./denoise/
python training_TF.py
```
* Modify `training_TF.py` 
  modeltypelst = ['FBuncertainty_f_twostage']
Replacing "modelpath1" with the path of the model trained in the first stage. 


#### Prediction

- Testing the model for denoising

```
cd ./denoise/
python training_TF.py
```
* Modify `training_TF.py` 
  istrain=False
  modeltypelst = ['FBuncertainty_f_twostage']
Replacing "testGTpath" with the path of the grount-truth images in the testing data.
Replacing "modelpath1" with the path of the model trained in the first stage. 
Replacing "modelpath" with the path of the model trained in the second stage.


### Data
All training and test data involved in the experiments are publicly available datasets and can be downloaded from `https://publications.mpi-cbg.de/publications-sites/7207/`

### Model
The pretrained models can be downloaded from `https://pan.baidu.com/s/1IqedS62A02THpRdvgyJEMg?pwd=rvcf`
