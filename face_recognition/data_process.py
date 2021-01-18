import numpy as np
import csv
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder


def csv_to_list(file_path):
    # convert csv to x_list,y_list
    x_list = []
    y_list = []
    x_tmp = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            x = row[0].strip('[').strip(']').split()
            y = row[1]
            if len(x) == 128:
                x_tmp.clear()
                for element in x:
                    x_tmp.append(float(element))
                x_list.append(x_tmp)
                y_list.append(int(y))
        return x_list, y_list


def list_to_dict(x_list1, y_list1, x_list2, y_list2):
    data_dict = {
        'images_train': np.array(x_list1),
        'labels_train': np.array(y_list1),
        'images_test': np.array(x_list2),
        'labels_test': np.array(y_list2),
    }
    return data_dict


def csv_to_dict(train_path, test_path):
    x_train, y_train = csv_to_list(train_path)
    x_test, y_test = csv_to_list(test_path)

    return list_to_dict(x_train, y_train, x_test, y_test)


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
            x.append(face_enc)
            y.append(i-1)
    return x, y

def load_test_file(i):
    x=[]
    y=[]
    for img in image_files_in_folder("./test_data/" + 'faces' + str(i)):
        face = face_recognition.load_image_file(img)
        face_bounding_boxes = face_recognition.face_locations(face)
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]
            x.append(face_enc)
            y.append(i-1)
    return x, y

def load_data():
    k = 7
    xs = []
    ys = []
    for i in range(1, k):
        print("Processing training data set " + str(i))
        X, Y = load_file(i)
        xs.append(X)
        ys.append(Y)
        print("Processing training data set " + str(i)+' finish!')
    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    del xs, ys

    xs = []
    ys = []
    for i in range(1, k):
        print("Processing test_data set " + str(i))
        X, Y = load_test_file(i)
        xs.append(X)
        ys.append(Y)
        print("Processing test data set " + str(i) + ' finish!')
    x_test = np.concatenate(xs)
    y_test = np.concatenate(ys)
    del xs, ys

    data_dict = {
        'images_train': x_train,
        'labels_train': y_train,
        'images_test': x_test,
        'labels_test': y_test,
    }
    return data_dict
