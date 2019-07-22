#-*- coding:UTF-8 -*-
import numpy as np
import os
import face_recognition

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
def face_distance1(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.

    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    dis=face_distance1(known_face_encodings, face_encoding_to_check)
    # print dis
    return list(face_distance1(known_face_encodings, face_encoding_to_check)<= tolerance),dis

def face(frame,known_faces,tolerance=0.5):
  match=[]
  d=[]
  face_encodings = face_recognition.face_encodings(frame)

  for face_encoding in face_encodings:
    match ,d= compare_faces(known_faces,face_encoding,tolerance=tolerance)
    # print match
  return match,d

def read_trainpic(path):
    known_faces=[]
    Name=[]


    for file in os.listdir(path):
        im_path=path+file
        name = file.split('.', 1)
        Name.append(name[0])
        lmm_image = face_recognition.load_image_file(im_path)
        face_encoding= face_recognition.face_encodings(lmm_image)[0]
        known_faces.append(face_encoding)
    return Name,known_faces