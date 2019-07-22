import os, glob
import cv2
from keras.utils import to_categorical
import numpy as np

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def load_retrain_data(parent_dir,file_ext='*.png', image_size = None, channel=3, crop_face = False):
    sub_dirs = os.listdir(parent_dir)
    ###########排列为标准顺序EMOTIONS = ["angry" ,"disgust","fear", "happy", "sad", "surprised", "neutral"]############
    sub_dirs.sort()
    if len(sub_dirs) == 7:
        sub_dirs[5], sub_dirs[4] = sub_dirs[4], sub_dirs[5]
        sub_dirs[6], sub_dirs[5] = sub_dirs[5], sub_dirs[6]
    face_data = []
    labels = []

    for label, sub_dir in enumerate(sub_dirs):
        if os.path.isdir(os.path.join(parent_dir, sub_dir)):
            for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                if channel==1:
                    image = cv2.imread(fn, 0)
                else:
                    image = cv2.imread(fn)
                if crop_face:
                    image = detect_face(image)
                image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
                # cv2.imshow('c', image)
                # cv2.waitKey(0)
                face_data.append(image)
                labels.append(label)
    labels_onehot = to_categorical(labels)
    labels_onehot = labels_onehot.astype(np.uint8)
    face_data1 = np.array(face_data)
    if face_data1.ndim == 3:
        face_data1 = np.expand_dims(face_data1, axis=-1)
        face_data1 = preprocess_input(face_data1)

    return face_data1, labels_onehot


def detect_face(image):
    detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
    face_detection = cv2.CascadeClassifier(detection_model_path)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) > 0:
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
        # the ROI for classification via the CNN
        roi_image = image[fY:fY + fH, fX:fX + fW]
        # cv2.imshow('222', roi_image)
        # cv2.waitKey(0)
        # roi_image = cv2.resize(roi_image, (48, 48))
        return roi_image
    else:
        return image