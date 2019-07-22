#-*- coding:UTF-8 -*-
from keras.preprocessing.image import img_to_array
import imutils
import cv2
import glob
from keras.models import load_model
import keras.backend as K
import time
#from models.cnn import resize
import tensorflow as tf
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
# import tensorflow as tf
import zipfile
import time
import cv2
import kcftracker
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import face_recognition
import collections
from utils import label_map_util
from utils import visualization_utils_color as vis_util
from function1 import *
from keras.applications.mobilenet import relu6

# input_shape = (181, 143, 3)
input_shape = (48, 48, 1)
sys.path.append("..")
emotion_model_path = 'models/best_model/MUL_KSIZE_MobileNet_v2_best.hdf5'

# K.clear_session()
EMOTIONS = ["Angry" ,"Disgust","Fear", "Happy", "Sad", "Surprise", "Neutral"]
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './MSKCF_model/frozen_inference_graph_face.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'
cv2.namedWindow('Main_window', cv2.WINDOW_AUTOSIZE)
# cv2.namedWindow('Main_window', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Main_window', 640, 480)

colors=[
     'Chartreuse', 'AliceBlue','Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=2, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Size, in inches, of the output images.
# IMAGE_SIZE = (12, 8)
# duration = 0.0
# initTrack=False
# onTracking=False

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("./video/10.mp4")
# cap = cv2.VideoCapture("./video/00.avi")
cap.set(3,640) #设置分辨率
cap.set(4,480)
# fps = cap.get(cv2.CAP_PROP_FPS)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('999.avi',fourcc, 20.0, (640,480))

dict1=collections.defaultdict(list)
dict2=collections.defaultdict(list)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Name,knows_face=read_trainpic('/home/prl/YOLO/facedata/')
with detection_graph.as_default():
  config = tf.ConfigProto(log_device_placement=True)
  config.gpu_options.allow_growth=True
  with tf.Session(graph=detection_graph, config=config) as sess:
    # frame_num = 1490;
    num=0
    duration1 = 0
    # tracker = kcftracker.KCFTracker(True, True, True)  # hog, fixed_window, multiscale
    aa = 0
    # fps = 0
    fps_counter = 0
    timer = time.time()
    frames = 0
    trackerm = []

    # Name, known_faces = read_trainpic('./facedata/')
    # emotion_classifier = load_model(emotion_model_path)
    emotion_classifier = load_model(emotion_model_path, custom_objects={'tf': tf, 'relu6': relu6})
    emotion_classifier.summary()
    save = False
    while( cap.isOpened()):

      ret, image = cap.read()
      # image = cv2.imread('/home/prl/dataset/SFEW_2/Train/Angry/Bridesmaids_012317840_00000001.png')
      cv2.imshow('0',image)

      start_time = time.time()
      # if ret == True:
      #   image = cv2.resize(image,(640,480))
      num+=1

      if save == True:
          cv2.imwrite('{}'.format(num) + '_0.png', image)

      if(ret ==0):
          break
      ###############################################################################
      # canvas = np.zeros((250, 300, 3), dtype="uint8")
      ########################################################################Tracking
      if ((num %10== 0) | (len(dict1) == 0)):

          trackers = []
          ix = []
          iy = []
          iw = []
          ih = []

          image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')


          # start_time = time.time()

          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})

#     min_score_thresh   人脸检测阈值 越大越严格
          count1,dict1=vis_util.visualize_boxes_and_labels_on_image_array(
    #          image_np,
              image,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),

              category_index,
              colors=colors,
              min_score_thresh=0.6,
              use_normalized_coordinates=True,
              line_thickness=4)
          for key, value in dict1.items():
                ix.append(dict1[key][0][3])
                iy.append(dict1[key][0][0])
                iw.append(dict1[key][0][1]-dict1[key][0][3])
                ih.append(dict1[key][0][2]-dict1[key][0][0])

          for x,y,w,h in zip(ix,iy,iw,ih):

              # tracker = cv2.Tracker_create("KCF")
              tracker1 = kcftracker.KCFTracker(False, True, True)  # hog, fixed_window, multiscale

              bbox = (x, y, w, h)

              tracker1.init(bbox, image)

              trackers.append(tracker1)

      bboxm=[]
      for tracker1 in trackers:
          bbox = tracker1.update(image)

          bboxm.append(bbox)

      bboxm = np.array(bboxm)
      facesize=0

      for box_x, box_y, box_w, box_h in bboxm:  # try boxm[ok==True][0] for box_x ....
          facesize+=1
          p1 = (int(box_x), int(box_y))
          p2 = (int(box_x + box_w), int(box_y + box_h))
          a = 10
          # image_cat=image[int(box_y-5-a):int(box_y+box_h+10+a),int(box_x-a):int(box_x+box_w+a)]
          image_cat=image[int(box_y):int(box_y+box_h),int(box_x):int(box_x+box_w)]
          cv2.imshow('image_cat', image_cat)
          # cv2.imwrite("1.jpg",image_cat)
          # cv2.waitKey()
          if save == True:
              cv2.imwrite('{}'.format(num)+'_1.png', image_cat)

          if(image_cat.shape[0] >= 20 and image_cat.shape[1] >= 20):
              ######################################################识别出人,很慢#############################
              # match,dis=face(image_cat,known_faces,0.005)
              # # print match
              # name = "Face"
              # # i=0
              # # for i in match:
              # #     if i==True:
              # #         i+=1
              #
              # # if i!=0:
              # if match:
              #     name=Name[np.argmin(dis)]
              # cv2.putText(image, name, (int(box_x),int(box_y-5) ), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)


##########################################################################################################

              # frameClone = image.copy()
              # if len(faces) > 0:
              # faces = sorted(image_cat, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
              # (fX, fY, fW, fH) = faces
              # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
              # the ROI for classification via the CNN
              # gray=[]
              # if image_cat.shape[-1]==3 and image_cat is not None:
              if input_shape[-1] == 1:
                image_cat = cv2.cvtColor(image_cat, cv2.COLOR_BGR2GRAY)
              roi = cv2.resize(image_cat, (input_shape[1], input_shape[0]))
              roi = roi.astype("float") / 255.0
              roi = img_to_array(roi)
              roi = np.expand_dims(roi, axis=0)

              preds = emotion_classifier.predict(roi)
              preds=preds[0]
              emotion_probability = np.max(preds)
              label = EMOTIONS[preds.argmax()]

              for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                  # construct the label text
                  text = "{}: {:.2f}%".format(emotion, prob * 100)

                  # draw the label + probability bar on the canvas
                  # emoji_face = feelings_faces[np.argmax(preds)]

                  w = int(prob * 150)
                  aa = 25
                  cv2.rectangle(image, (1, (i * aa) + 5),
                                (w, (i * aa) + aa), (0, 0, 255), -1)
                  cv2.putText(image, text, (10, (i * aa) + 18),
                              cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                              ( 255,0, 0), 1)

                  # cv2.rectangle(canvas, (7, (i * 35) + 5),
                  #               (w, (i * 35) + 35), (0, 0, 255), -1)
                  # cv2.putText(canvas, text, (10, (i * 35) + 23),
                  #             cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                  #             (0, 255, 0), 1)

                  cv2.putText(image, label+' ('+str('%.2f' % emotion_probability)+')', (p1[0], p1[1]-10),
                              cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
                  cv2.rectangle(image, p1, p2, (0, 0, 255), 2)
####################################################################################################
      end = time.time()
      seconds = end - start_time
      if seconds==0:
          seconds=0
      else:
          fps = 1 / seconds
      # fps = 78
      cv2.putText(image, "FPS: " + str('%.0f' % fps), (560, 18), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,0,  0),
                  1, False)
      duration1 = duration1 + seconds
      if num % 500 == 0:
          print(int(500 / duration1))
          duration1 = 0
#####################################################################################################

      cv2.imshow('Main_window', image)
      # cv2.imshow("Probabilities", canvas)
      # out.write(image)
      if save == True:
          cv2.imwrite('{}'.format(num)+'_2.png', image)

      save = False
      if cv2.waitKey(1) & 0xFF == ord('s'):
          save = True
          # cv2.imwrite('0.png',image)
          # cv2.imwrite('1.png', image111)
          # cv2.imwrite('2.png', image_cat111)

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    # out.release()
    cap.release()
    cv2.destroyAllWindows()