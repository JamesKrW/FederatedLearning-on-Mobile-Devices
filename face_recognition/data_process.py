import numpy as np
import csv
import os
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder


def csv_to_list(file_path):
    # convert csv to x_list,y_list
    x_list = []
    y_list = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            x_tmp = []
            x = row[0].strip('[').strip(']').split()
            y = row[1]
            if len(x) == 128:
                for element in x:
                    x_tmp.append(float(element))
                x_tmp = np.array(x_tmp)
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


def load_file(i, name_label):
    x=[]
    y=[]
    for img in image_files_in_folder("./train_data/" + i):
        face = face_recognition.load_image_file(img)
        face_bounding_boxes = face_recognition.face_locations(face)
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]
            x.append(face_enc)
            y.append(name_label[i])
    return x, y

def load_test_file(i, name_label):
    x=[]
    y=[]
    for img in image_files_in_folder("./test_data/" + i):
        face = face_recognition.load_image_file(img)
        face_bounding_boxes = face_recognition.face_locations(face)
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]
            x.append(face_enc)
            y.append(name_label[i])
    return x, y

def load_data(name_label):
    files = []
    for root in os.listdir('./train_data'):
        if root != '.DS_Store':
            files.append(root)
    k = len(files)
    xs = []
    ys = []
    for i in range(1, k):
        print("Processing train data set " + files[i])
        X, Y = load_file(files[i], name_label)
        xs.append(X)
        ys.append(Y)
        print("Processing train data set " + files[i] + ' finish!')
    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    del xs, ys

    files = []
    for root in os.listdir('./test_data'):
        if root != '.DS_Store':
            files.append(root)
    k = len(files)
    xs = []
    ys = []
    for i in range(1, k):
        print("Processing test_data set " + files[i])
        X, Y = load_test_file(files[i], name_label)
        xs.append(X)
        ys.append(Y)
        print("Processing test data set " + files[i] + ' finish!')
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

def load_data_csv(name_label):
    xs = []
    ys = []
    path = './train_data_csv'
    for dir in os.listdir(path):
        dirpath = os.path.join(path, dir)
        if os.path.isdir(dirpath) and dir != '.DS_Store':
            for file in os.listdir(dirpath):
                filepath = os.path.join(dirpath, file)
                x = np.loadtxt(open(filepath, "rb"), delimiter=",", skiprows=0)
                print(dir)
                y = name_label[dir]
                xs.append(x)
                ys.append(y)
    x_train = np.array(xs)
    y_train = np.array(ys)
    del xs, ys

    xs = []
    ys = []
    print('Begin to process test data set')
    path = './test_data_csv'
    for dir in os.listdir(path):
        dirpath = os.path.join(path, dir)
        if os.path.isdir(dirpath) and dir != '.DS_Store':
            for file in os.listdir(dirpath):
                filepath = os.path.join(dirpath, file)
                x = np.loadtxt(open(filepath, "rb"), delimiter=",", skiprows=0)
                print(dir)
                y = name_label[dir]
                xs.append(x)
                ys.append(y)
    x_test = np.array(xs)
    y_test = np.array(ys)
    del xs, ys

    data_dict = {
        'images_train': x_train,
        'labels_train': y_train,
        'images_test': x_test,
        'labels_test': y_test,
    }
    return data_dict