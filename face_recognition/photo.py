import cv2
import tensorflow as tf
import os
import numpy as np
import sys
import face_recognition
from speak import speak
import time

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
persons = 6
names = {4: '王康睿', 1:'谭杨汶坚', 3:'陈泽越', 2:'Others', 5:'曾航', 6:'张远航'}

# Define input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, 128])
labels_placeholder = tf.placeholder(tf.int64, shape=[None])

# Define variables
weightsl1 = tf.Variable(tf.random_normal([128, 60]))
biasesl1 = tf.Variable(tf.random_normal([60]))
weightsl2 = tf.Variable(tf.zeros([60, persons]))
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
            out = sess.run(tf.argmax(net, 1), feed_dict={images_placeholder: face_enc})
            out = out[0]
            print('time:{}, start_time:{}'.format(time.time(), start_time))
            if last != out or time.time()-start_time>30:
                speak(names[out + 1])
                start_time = time.time()
            last = out
            print(names[out + 1])
