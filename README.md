# Real-time-Facial-Expression-Recognition-and-Fast-Face-Detection
Real-time facial expression recognition and fast face detection based on Keras CNN. Training and testing on both Fer2013 and CK+ facial expression data sets have achieved good results. The speed is 78 fps on NVIDIA 1080Ti. If only face detection is performed, the speed can reach 158 fps. Finally, an emotional monitoring system was developed based on it.

## Dependencies
* Ubuntu 16.04
* Tensorflow 1.8.0
* Keras 2.1.6
* Pycharm 2018
* Python 3.6
* PyQt5
* Numpy 1.14.5
* Pandas 0.24.2
* Sklearn 0.21.0

Note: Keras must be installed under v2.2, because some functions in v2.2 are different from V2.1, and individual functions do not exist in v2.2.

## Examples
* Face detection：

<img src="https://github.com/liminze/Real-time-Facial-Expression-Recognition-and-Fast-Face-Detection/blob/master/images/1.png" width="320" height="240" />

* Facial expression recognition:

<img src="https://github.com/liminze/Real-time-Facial-Expression-Recognition-and-Fast-Face-Detection/blob/master/images/2.png" width="430" height="320" />

* Emotional monitoring system:

<img src="https://github.com/liminze/Real-time-Facial-Expression-Recognition-and-Fast-Face-Detection/blob/master/images/3.png" width="430" height="320" /> <img src="https://github.com/liminze/Real-time-Facial-Expression-Recognition-and-Fast-Face-Detection/blob/master/images/4.png" width="430" height="320" />

<img src="https://github.com/liminze/Real-time-Facial-Expression-Recognition-and-Fast-Face-Detection/blob/master/images/5.png" width="430" height="320" /> <img src="https://github.com/liminze/Real-time-Facial-Expression-Recognition-and-Fast-Face-Detection/blob/master/images/6.png" width="430" height="320" />

<img src="https://github.com/liminze/Real-time-Facial-Expression-Recognition-and-Fast-Face-Detection/blob/master/images/7.png" width="430" height="320" /> <img src="https://github.com/liminze/Real-time-Facial-Expression-Recognition-and-Fast-Face-Detection/blob/master/images/8.png" width="430" height="320" />

<img src="https://github.com/liminze/Real-time-Facial-Expression-Recognition-and-Fast-Face-Detection/blob/master/images/9.png" width="430" height="320" /> <img src="https://github.com/liminze/Real-time-Facial-Expression-Recognition-and-Fast-Face-Detection/blob/master/images/10.png" width="430" height="320" />

* Confusion matrix ( left - fer2013, right - CK+ ):

<img src="https://github.com/liminze/Real-time-Facial-Expression-Recognition-and-Fast-Face-Detection/blob/master/images/fer2013.png" width="430" height="320" /> <img src="https://github.com/liminze/Real-time-Facial-Expression-Recognition-and-Fast-Face-Detection/blob/master/images/CK+.png" width="430" height="320" />

## Fer2013 Dataset
* Properties: 48 x 48 pixels (2304 bytes) labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral. The training set consists of 28,709 examples. The public test set consists of 3,589 examples. The private test set consists of another 3,589 examples.

* Path: "fer2013/fer2013.csv". If it does not exist, you can download from the link. https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

## CK+ Dataset
* The CK+ facial expression dataset consists of 123 individuals, 593 image sequences, and the last one of each image sequence has an action unit label, and 327 of the image sequences have an emoticon label, which is labeled as seven types of emoticons: anger, contempt, disgust, fear, happy, sadness, and surprise. Other expression datasets do not have contempt expression, so in order to be compatible with other data sets, the expressions are despised.

* Path:  "other_dataset".

## Trained Models
* [frozen_inference_graph_face.pb]: Use this model for face detection, the path is “MSKCF_model / frozen_inference_graph_face.pb”.

* [MUL_KSIZE_MobileNet_v2_best.hdf5]: Use it for facial expression recognition, the path is “models / best_model / MUL_KSIZE_MobileNet_v2_best.hdf5”.

## How to use it
Note: Make sure the camera is turned on before use and the path to the model is correct.

* Run MS_FER_inference.py.

Fast facial expression recognition (face detection using Mobilenet-SSD+KCF).

* Run real_time_video(old).py.

Normal facial expression recognition (face detection using Haar-cascade in OpenCV).

* Run ysdui.py.

Opening emotional monitoring system. Emotion monitoring can be done by clicking the start button on the interface.

* Run train_emotion_classifier.py.

Training expression recognition model.


## Retraining other dataset
* Run train_again_emotion.py.

Note: Put the dataset that needs to be trained again into the "other_dataset" folder (I have placed it in the CK+ dataset). The format of the retrained data should be consistent with the existing CK+ format. In addition, the CK+ dataset uses only 6 classes, while the original model is 7 classes, so there is an operation to rebuild the network. If the classes are the same, you can remove the action.

## Plot confusion matrix
* Run plot_confusion_mat.py. 

Note: The model loading path and number of tag categories must is right.






