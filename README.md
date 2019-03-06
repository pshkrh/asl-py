# ASL Classification

This is an American Sign Language Classifier, built using fastai v1 (based on PyTorch), and OpenCV. 
Trained on the ResNet34 Pre-Trained model.

## Dataset

The dataset can be found here:

```
https://www.kaggle.com/grassknoted/asl-alphabet
```

## Requirements

This project uses Python 3 (specifically 3.6) and requires the following packages:

* OpenCV
* fastai
* matplotlib
* numpy

For OpenCV: 

```
sudo apt-get install python-opencv
```

For the Python Packages:

### Installation using pip
```
pip3 install -r requirements.txt
```

## Training 

To train the model, simply use the following command:

```
python3 train.py
```
Note: GPU is recommended.

## Usage

Once the model has been trained, or if you are using the provided export.pkl (model) file, use the following command:

```
python3 webcam.py
```
After 4-5 seconds, a window should start up with the webcam. Perform your ASL alphabet signs in the provided blue box (the region of interest).

Hold the sign to repeat it, or change it. You will be able to start typing right away.
