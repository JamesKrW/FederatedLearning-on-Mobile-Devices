import cv2
import tensorflow as tf
import os
import numpy as np
import face_recognition
import speak as sb
import time
import socket
import requests
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
persons = 100


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
name_dict2 = dict(zip(name_dict.values(), name_dict.keys()))
print(name_dict2)
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
            pr = sess.run(tf.nn.softmax(net), feed_dict={images_placeholder: face_enc})
            out = sess.run(tf.argmax(pr, 1))
            out = out[0]
            print("out:", out)
            print("pr:", pr)
            print('time:{}, start_time:{}'.format(time.time(), start_time))
            if last != out or time.time()-start_time>30:
                if (pr[0][out] < 0.333):
                    sb.speak("hi, stranger!")
                    print("stranger:")
                    print(name_dict[str(out)])
                else:
                    sb.speak(name_dict[str(out)])
                    start_time = time.time()
                    last = out
                    print(name_dict[str(out)])
