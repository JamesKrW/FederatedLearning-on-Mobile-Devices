import numpy as np
import pandas as pd
import gc
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder


person = 6
def load_file(i):
    x=[]
    y=[]
    for img in image_files_in_folder("./train_data/" + 'faces' + str(i)):
        '''
        print(img)
        '''
        face = face_recognition.load_image_file(img)
        face_bounding_boxes = face_recognition.face_locations(face)
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]
            x = x + [face_enc]
            y = y + [i-1]
    return x, y

def load_test_file(i):
    x=[]
    y=[]
    for img in image_files_in_folder("./test_data/" + 'faces' + str(i)):
        face = face_recognition.load_image_file(img)
        face_bounding_boxes = face_recognition.face_locations(face)
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]
            x = x + [face_enc]
            y = y + [i-1]
    return x, y


def save_csv():
    k = person + 1
    xs = []
    ys = []
    for i in range(1, k):
        print("Processing training data set " + str(i))
        X, Y = load_file(i)
        xs = xs + X
        ys = ys + Y
        print("Processing training data set " + str(i)+' finish!')
    df = pd.DataFrame.from_dict({'data': xs, 'label': ys})
    df.to_csv('./train' + '.csv', header=True, index=False, columns=['data', 'label'])
    gc.collect()
    del xs, ys

    xs = []
    ys = []
    for i in range(1, k):
        print("Processing test_data set " + str(i))
        X, Y = load_test_file(i)
        xs = xs + X
        ys = ys + Y
        print("Processing test data set " + str(i) + ' finish!')
    df = pd.DataFrame.from_dict({'data': xs, 'label': ys})
    df.to_csv('./test' + '.csv', header=True, index=False, columns=['data', 'label'])
    gc.collect()
    del xs, ys
    return

