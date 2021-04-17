import cv2
import tensorflow as tf
import os
import numpy as np
import sys
import face_recognition
from speak import speak
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#This should be alternated to adapt the new model
persons = 100

import socket  # 导入 socket 模块
s = socket.socket()  # 创建 socket 对象
host = '0.0.0.0'  # 获取本地主机名
port = 12350  # 设置端口
s.bind((host, port))  # 绑定端口
s.listen(2)  # 等待客户端连接

def getmessage():
    c, addr = s.accept()  # 建立客户端连接
    print('连接地址', addr)
    print('send over')
    while True:
        data = c.recv(1024)
        if data:
            ans = data.decode()
            print("message:", ans)
            break
    c.close()  # 关闭连接
    return ans

name_list = []
label_list = []
f=open('namelabel.csv', 'r')
next(f)
lines = f.readlines()
f.close()
for line in lines:
    name=line.strip().split(',')[0].strip()
    label=line.strip().split(',')[1].strip()
    name_list.append(name)
    label_list.append(label)
name_dict={}
for i in range(len(name_list)):
    name_dict[label_list[i]]=name_list[i]

# Define input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, 128])
labels_placeholder = tf.placeholder(tf.int64, shape=[None])

# Define variables
weightsl1 = tf.Variable(tf.random_normal([128, 256]))
biasesl1 = tf.Variable(tf.random_normal([256]))
weightsl2 = tf.Variable(tf.zeros([256, persons]))
biasesl2 = tf.Variable(tf.zeros([persons]))

# Define net
net = images_placeholder
net = tf.nn.relu(tf.add(tf.matmul(net, weightsl1), biasesl1))
net = tf.add(tf.matmul(net, weightsl2), biasesl2)

saver = tf.train.Saver()
if not os.path.exists('./tmp/'):
    os.mkdir('./tmp/')

with tf.Session() as sess:
    if os.path.exists('./tmp/checkpoint'):
        saver.restore(sess, './tmp/model.ckpt')
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
    if not os.path.exists('./photos/'):
        os.mkdir('./photos')
    addr = './photos/'
    index = 0
    last = 999
    start_time = time.time()
    while 1:
        cap = cv2.VideoCapture("rtsp://admin:gosunyun888@10.7.5.221:554/h264/ch1/main/av_stream")
        print(index)
        success, frame = cap.read()
        addr = './photos/photo' + str(index) + '.jpg'
        index = index + 1
        cv2.imwrite(addr, frame)
        face = face_recognition.load_image_file(addr)
        face_bounding_boxes = face_recognition.face_locations(face)
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]
            face_enc = np.reshape(face_enc, [1, 128])
            pr = sess.run(tf.nn.softmax(net),feed_dict={images_placeholder: face_enc})
            out = sess.run(tf.argmax(pr, 1))
            out = out[0]
            print("out:",out)
            print("pr:",pr) 
            print('time:{}, start_time:{}'.format(time.time(), start_time))
            if last != out or time.time()-start_time>30:
                if (pr[0][out] < 0.5):
                    speak("hi, please register your name")
                    name=getmessage()
                    path="./train_data/name"
                    if not os.path.exists(path):
                        os.makedirs(path)
                    for i in range(100):
                        cv2.imwrite(os.path.join(path, str(i) + '.jpg'), face)
                else:
                    speak(name_dict[out])
                    start_time = time.time()
            last = out
            print(name_dict[out])
