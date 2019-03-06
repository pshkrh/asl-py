# ASL Classification

This is an American Sign Language Classifier, built using fastai v1 (based on PyTorch), and OpenCV. 
Trained on the ResNet34 Pre-Trained model.

## Demo

![ASL Classifier Demo](https://i.imgur.com/gesI2thh.gif)

Full Video can be found here: https://youtu.be/jKA2dMz0bP8

## Dataset

The dataset can be found here:

```
https://www.kaggle.com/grassknoted/asl-alphabet
```
Note: The dataset is large, approximately 1GB.

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

```
pip3 install -r requirements.txt
```

## Training 

Open up the train.py file, and point the path variable to the path of the dataset on your computer.
Then, to train the model, simply use the following command:

```
python3 train.py
```
Note: A good GPU is recommended for training. Training can take up to or even more than an hour.

## Usage

Open up the webcam.py file, and change the path variable to point to the export.pkl file, and change the imgpath variable to point to where you would like the saved frame to be stored.
Once the model has been trained, or if you are using the provided export.pkl (model) file, use the following command:

```
python3 webcam.py
```
After 4-5 seconds, a window should start up with the webcam. Perform your ASL alphabet signs in the provided blue box (the region of interest).

Hold the sign to repeat it, or change it. You will be able to start typing right away.
