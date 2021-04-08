import numpy as np
import tensorflow as tf
import os
import sys
import face_recognition


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
learning_rate = 0.001
epoch = 5000
persons = 100
names = {3: '王康睿', 0:'谭杨汶坚', 2:'陈泽越', 1:'wjl', 4:'曾航', 5:'张远航'}

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
net = tf.nn.softmax(net)

# Define loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=labels_placeholder))

# Define operation
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(net, 1), labels_placeholder)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    if os.path.exists('./5_people_model/checkpoint'):
        saver.restore(sess, './5_people_model/model.ckpt')
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    face = face_recognition.load_image_file('./test_data/tywj/51.jpg')
    face_bounding_boxes = face_recognition.face_locations(face)
    if len(face_bounding_boxes) == 1:
        face_enc = face_recognition.face_encodings(face)[0]
        face_enc = np.reshape(face_enc, [1, 128])
        print(sess.run(net, feed_dict={images_placeholder: face_enc}))
        pr = sess.run(net, feed_dict={images_placeholder: face_enc})
        out = sess.run(tf.argmax(pr, 1))
        out = out[0]
        print(pr[0, out])
        print(names[out])


