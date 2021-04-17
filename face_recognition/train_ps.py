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


PS_PRIVATE_IP = "0.0.0.0:61234"
PS_PUBLIC_IP = "0.0.0.0:61234"

persons = 100
communication_rounds = 50
local_iter_num = 100
train_batch_size = 32
learning_rate = 0.0001
decay_rate = 0.95

# Define input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, 128])
labels_placeholder = tf.placeholder(tf.int64, shape=[None])

# Define variables
weightsl1 = tf.Variable(tf.random_normal([128,256 ]))
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

sess = tf.Session()
if os.path.exists('./tmp/checkpoint'):
    saver.restore(sess, './tmp/model.ckpt')
else:
    init = tf.global_variables_initializer()
    sess.run(init)

#------------------------receive namelist-----------------------------------
communication = Communication(PS_PRIVATE_IP, PS_PUBLIC_IP)
ps_socket= communication.start_socket_ps()
c_1, _ = ps_socket.accept()
c_2, _ = ps_socket.accept()
#------------------------device1-----------------------------------
received_message_1 = pickle.loads(communication.get_message(c_1))
#------------------------device2-----------------------------------
received_message_2 = pickle.loads(communication.get_message(c_2))
#------------------------process name label-----------------------------------
name_label_1 = received_message_1
name_label_2 = received_message_2
name_list=[]
label_list=[]
f=open('namelabel.csv', 'r')
next(f)
lines = f.readlines()
f.close()
for line in lines:
    name=line.strip().split(',')[0].strip()
    label=line.strip().split(',')[1].strip()
    name_list.append(name)
    label_list.append(label)
addon1=list(set(name_label_1)-set(name_list))
addon2=list(set(name_label_2)-set(name_list))
for name in addon1:
    name_list.append(name)
    x=len(label_list)
    label_list.append(str(x))
for name in addon2:
    name_list.append(name)
    x=len(label_list)
    label_list.append(str(x))
name_dict={}
f=open('namelabel.csv', 'a')
for i in range(len(name_list)):
    name_dict[name_list[i]]=label_list[i]
    f.write(name_list[i]+','+label_list[i])
f.close()

#-----------------------------------------------------------
hyperparameters = {'communication_rounds': communication_rounds,
                   'local_iter_num': local_iter_num,
                   'train_batch_size': train_batch_size,
                   'learning_rate': learning_rate,
                   'decay_rate': decay_rate,
                   'namelabel':name_dict,
                   'persons':persons}

model_paras = sess.run(tf.trainable_variables())

print("ready for connection")

#------------------------send message to device1-----------------------------------
send_message = {'model_paras': model_paras, 'hyperparameters': hyperparameters}
communication.send_message(pickle.dumps(send_message), c_1)

print("device1 sent ")
#------------------------send message to device2-----------------------------------
send_message = {'model_paras': model_paras, 'hyperparameters': hyperparameters}
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
ps_socket.close()


#------------------------------------------------------------------------multi-thread
