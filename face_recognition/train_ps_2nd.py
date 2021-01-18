import tensorflow as tf
import os
import numpy as np
from communication import Communication
import pickle
from general_functions import *

PS_PRIVATE_IP = "0.0.0.0:61234"
PS_PUBLIC_IP = "0.0.0.0:61234"
persons = 6
communication_rounds = 2
local_iter_num = 1000
train_batch_size = 32
learning_rate = 0.001
decay_rate = 0.95

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

sess = tf.Session()
if os.path.exists('./tmp/checkpoint'):
    saver.restore(sess, './tmp/model.ckpt')
else:
    init = tf.global_variables_initializer()
    sess.run(init)

hyperparameters = {'communication_rounds': communication_rounds,
                   'local_iter_num': local_iter_num,
                   'train_batch_size': train_batch_size,
                   'learning_rate': learning_rate,
                   'decay_rate': decay_rate}

model_paras = sess.run(tf.trainable_variables())
send_message = {'model_paras': model_paras, 'hyperparameters': hyperparameters}
communication = Communication(PS_PRIVATE_IP, PS_PUBLIC_IP)
ps_socket = communication.start_socket_ps()
c, _ = ps_socket.accept()
communication.send_message(pickle.dumps(send_message), c)

for i in range(communication_rounds):
    print('begin get')
    received_message = pickle.loads(communication.get_message(c))
    #print('receive:', received_message)
    print('get over')
    delta_model_paras = received_message['model_paras']
    new_model_paras = [np.zeros(weights.shape) for weights in model_paras]
    for index in range(len(model_paras)):
        new_model_paras[index] = model_paras[index] + delta_model_paras[index]
    placeholders = create_placeholders()
    feed_dict = {}
    for place, para in zip(placeholders, new_model_paras):
        feed_dict[place] = para
    update_local_vars_op = assign_vars(tf.trainable_variables(), placeholders)
    sess.run(update_local_vars_op, feed_dict=feed_dict)
    send_message = {'model_paras': new_model_paras}
    print('begin send')
    communication.send_message(pickle.dumps(send_message), c)
    print('send over')
    model_paras = new_model_paras

saver.save(sess, './tmp/model.ckpt')