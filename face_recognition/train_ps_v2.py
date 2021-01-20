import tensorflow as tf
import os
import numpy as np
from communication import Communication
import pickle


def create_placeholders():
    placeholders = []
    for var in tf.trainable_variables():
        placeholders.append(tf.placeholder_with_default(var, var.shape,
                                                        name="%s/%s" % ("FedAvg", var.op.name)))
    return placeholders


def assign_vars(local_vars, placeholders):
    reassign_ops = []
    for var, fvar in zip(local_vars, placeholders):
        reassign_ops.append(tf.assign(var, fvar))
    return tf.group(*(reassign_ops))


PS_PRIVATE_IP_1 = "10.162.83.55:61234"
PS_PUBLIC_IP_1 = "10.162.83.55:61234"
PS_PRIVATE_IP_2 = "10.162.83.55:61234"
PS_PUBLIC_IP_2 = "10.162.83.55:61234"
persons = 6
communication_rounds = 5
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

print("ready for connection")

#------------------------send message to device1-----------------------------------
send_message = {'model_paras': model_paras, 'hyperparameters': hyperparameters}
communication = Communication(PS_PRIVATE_IP_1, PS_PUBLIC_IP_1)
ps_socket_1 = communication.start_socket_ps()
c_1, _ = ps_socket_1.accept()
communication.send_message(pickle.dumps(send_message), c_1)

print("device1 sent ")
#------------------------send message to device2-----------------------------------
send_message = {'model_paras': model_paras, 'hyperparameters': hyperparameters}
'''
communication = Communication(PS_PRIVATE_IP_2, PS_PUBLIC_IP_2)
ps_socket_2 = communication.start_socket_ps()
'''
c_2, _ = ps_socket_1.accept()
communication.send_message(pickle.dumps(send_message), c_2)

print("device2 sent ")

for i in range(communication_rounds):
    print('device1 begin get')
    received_message_1 = pickle.loads(communication.get_message(c_1))
    # print('receive:', received_message)
    print('device1 get over')
    print('-----------------------------------------')

    print('device2 begin get')
    received_message_2 = pickle.loads(communication.get_message(c_2))
    # print('receive:', received_message)
    print('device2 get over')
    print('-----------------------------------------')

    delta_model_paras_1 = received_message_1['model_paras']
    delta_model_paras_2 = received_message_2['model_paras']

    new_model_paras = [np.zeros(weights.shape) for weights in model_paras]

    print('communication_rounds:',i)
    print(received_message_1['count'])
    print(received_message_2['count'])
    print('-------------------------------')

    coefficient_1=float(received_message_1['count'])/(received_message_1['count']+received_message_2['count'])
    coefficient_2=float(received_message_2['count'])/(received_message_1['count']+received_message_2['count'])

    for index in range(len(model_paras)):
        new_model_paras[index] = model_paras[index] + coefficient_1*delta_model_paras_1[index] + \
                                 coefficient_2*delta_model_paras_2[index]
    placeholders = create_placeholders()
    feed_dict = {}
    for place, para in zip(placeholders, new_model_paras):
        feed_dict[place] = para
    update_local_vars_op = assign_vars(tf.trainable_variables(), placeholders)
    sess.run(update_local_vars_op, feed_dict=feed_dict)
    send_message = {'model_paras': new_model_paras}
    if i != communication_rounds - 1:
        print('device1 begin send')
        communication.send_message(pickle.dumps(send_message), c_1)
        print('device1 send over')
        print('---------------------------------')
        print('device2 begin send')
        communication.send_message(pickle.dumps(send_message), c_2)
        print('device2 send over')
        print('---------------------------------')
    model_paras = new_model_paras
    saver.save(sess, './tmp/model.ckpt')
ps_socket_1.close()
ps_socket_2.close()

#------------------------------------------------------------------------multi-thread

