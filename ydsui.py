# -*- coding:UTF-8 -*-
from keras.preprocessing.image import img_to_array
import imutils
from keras.models import load_model
import numpy as np
import keras.backend as K
import time
import tensorflow as tf
from function1 import *

from yds import Ui_MainWindow
import sys
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
from PyQt5 import QtCore

detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/best_model/MUL_KSIZE_MobileNet_v2_best.hdf5'

K.clear_session()
EMOTIONS = ["愤怒" ,"厌恶","惊讶", "高兴", "悲伤", "惊讶", "自然"]
EMOTIONS_LEVEL = ["低" ,"中","中", "高", "低", "中", "高"]
EMOTIONS_INFO = ["道路千万条，安全第一条；\n行车不规范，亲人两行泪" ,
                 "开好自己的车",
                 "请集中注意力，\n注意行车安全",
                 "开车也是件心情愉悦的事",
                 "道路千万条，安全第一条；\n行车不规范，亲人两行泪",
                 "请集中注意力，\n注意行车安全",
                 "状态不错，\n继续保持"]
Name, known_faces = read_trainpic('./facedata/')


class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)

        self.timer_camera = QTimer()#定义定时器
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)  # 设置分辨率
        self.cap.set(4, 480)
        self.face_detection = cv2.CascadeClassifier(detection_model_path)
        self.emotion_classifier = load_model(emotion_model_path, compile=False, custom_objects={'tf': tf})


    def content(self):
        self.cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        self.timer_camera.start(0)
        self.timer_camera.timeout.connect(self.openFrame)

    def pause(self):
        self.cap.release()
        self.timer_camera.stop()  # 停止计时器
        self.label_2.clear()
        self.label.clear()


    def openFrame(self):
        # global label_R
        start = time.time()
        if (self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret:
                frame = imutils.resize(frame, width=300)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                        flags=cv2.CASCADE_SCALE_IMAGE)
                # canvas = np.zeros((250, 300, 3), dtype="uint8")
                frameClone = frame.copy()
                if len(faces) > 0:
                    faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                    (fX, fY, fW, fH) = faces

                    RR = frame[fY:fY + fH, fX:fX + fW]

                    match, dis = face(RR, known_faces, 0.05)
                    name = "Face"
                    if match:
                        name = Name[np.argmin(dis)]

                    self.label_3.setText(name)
                    RR = cv2.cvtColor(RR, cv2.COLOR_BGR2RGB)
                    height, width, bytesPerComponent = RR.shape
                    bytesPerLine = bytesPerComponent * width
                    q_image_r = QImage(RR.data, width, height, bytesPerLine,
                                     QImage.Format_RGB888).scaled(self.label_2.width(), self.label_2.height())
                    self.label_2.setPixmap(QPixmap.fromImage(q_image_r))

                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
                    # the ROI for classification via the CNN
                    roi = gray[fY:fY + fH, fX:fX + fW]
                    roi = cv2.resize(roi, (48, 48))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    preds = self.emotion_classifier.predict(roi)[0]
                    emotion_probability = np.max(preds)
                    label_R = EMOTIONS[preds.argmax()]
                    label_L = EMOTIONS_LEVEL[preds.argmax()]
                    label_I = EMOTIONS_INFO[preds.argmax()]
                    self.label_19.setText(label_R)
                    self.label_21.setText(label_L)
                    self.label_23.setText(label_I)
                    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                        # construct the label text
                        text = "{}: {:.2f}%".format(emotion, prob * 100)

                        # draw the label + probability bar on the canvas
                        # emoji_face = feelings_faces[np.argmax(preds)]

                        w = int(prob * 300)
                        # cv2.rectangle(canvas, (7, (i * 35) + 5),
                        #               (w, (i * 35) + 35), (0, 0, 255), -1)
                        # cv2.putText(canvas, text, (10, (i * 35) + 23),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        #             (255, 255, 255), 2)
                        # cv2.putText(frameClone, label_R, (fX, fY - 10),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                                      (0, 0, 255), 2)

                end = time.time()
                seconds = end - start
                fps = 1 / seconds
                # fps += 60
                self.lcdNumber.display(str('%.0f' % fps))
                # cv2.putText(frameClone, "FPS: " + str('%.0f' % fps), (5, 15), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                #             (0, 0, 255),
                #             1, False)
                frameClone = cv2.cvtColor(frameClone, cv2.COLOR_BGR2RGB)
                height, width, bytesPerComponent = frameClone.shape
                bytesPerLine = bytesPerComponent * width
                q_image = QImage(frameClone.data, width, height, bytesPerLine,
                                 QImage.Format_RGB888).scaled(self.label.width(), self.label.height())
                self.label.setPixmap(QPixmap.fromImage(q_image))
                # self.textEdit.clear()
                # self.textEdit.setPlainText(label_R)

            else:
                self.cap.release()
                self.timer_camera.stop()  # 停止计时器

if __name__ == '__main__':
    #"""Configuration for GPU"""
    from keras.backend.tensorflow_backend import set_session

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)  # 使用第一台GPU
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  # GPU使用率为50%
    config.gpu_options.allow_growth = True  # 允许容量增长
    set_session(tf.InteractiveSession(config=config))

    app = QtWidgets.QApplication(sys.argv)
    myshow = mywindow()
    # myshow.setStyleSheet("#MainWindow{border-image:url(black.jpg);}")
    myshow.show()
    sys.exit(app.exec_())