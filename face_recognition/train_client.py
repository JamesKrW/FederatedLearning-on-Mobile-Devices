import os
import pickle
import numpy as np
import tensorflow as tf
import sys
import time
from data_process import load_data
from communication import Communication


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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


np.random.seed(1234)
tf.set_random_seed(1234)
PS_PUBLIC_IP = '10.162.83.55:7777'  # Public IP of the ps
PS_PRIVATE_IP = '10.162.83.55:7777'  # Private IP of the ps

# Create the communication object and get the training hyperparameters
communication = Communication(False, PS_PRIVATE_IP, PS_PUBLIC_IP)
client_socket = communication.start_socket_client()
print('Waiting for PS\'s command...')
sys.stdout.flush()
client_socket.settimeout(300)

# hyperparameters = communication.get_np_array(client_socket)
received_message = pickle.loads(communication.get_message(client_socket))
hyperparameters = received_message['hyperparameters']
old_model_paras = received_message['model_paras']
communication_rounds = hyperparameters['communication_rounds']
local_epoch_num = hyperparameters['local_iter_num']
train_batch_size = hyperparameters['train_batch_size']
learning_rate = hyperparameters['learning_rate']
decay_rate = hyperparameters['decay_rate']

data_sets = load_data()
persons = 6

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

# Define loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=labels_placeholder))

# Define operation
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(net, 1), labels_placeholder)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for round_num in range(communication_rounds):
    tf.reset_default_graph()
    sess = tf.Session()

    # communicate with ps, send batches_info and receive current model
    # client_socket = communication.start_socket_client()
    print('Waiting for PS\'s command...')
    sys.stdout.flush()
    client_socket.settimeout(300)
    if round_num != 0:
        while True:
            received_message = communication.get_message(client_socket)
            received_dict = pickle.loads(received_message)
            old_model_paras = received_dict['model_paras']
            client_socket.close()
            print('Received model parameter.')
            sys.stdout.flush()
            break

    # create model and initialize it with the embedding and model parameters pulled from ps
    ''' 
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    model_para = tf.trainable_variables()
    print(sess.run(model_para))
    for v, para in zip(model_para, old_model_paras):
        sess.run(tf.assign(v, para))
    print(sess.run(model_para))
    '''
    placeholders = create_placeholders()
    feed_dict = {}
    for place, para in zip(placeholders, old_model_paras):
        feed_dict[place] = para
    update_local_vars_op = assign_vars(tf.trainable_variables(), placeholders)
    sess.run(update_local_vars_op, feed_dict=feed_dict)
    print('Weights succesfully initialized')
    sys.stdout.flush()
    # begin training process
    print('Begin training')
    sys.stdout.flush()
    start_time = time.time()

    best_acc = 0
    for i in range(local_epoch_num):
        indices = np.random.choice(data_sets['images_train'].shape[0], train_batch_size)
        images_batch = data_sets['images_train'][indices]
        labels_batch = data_sets['labels_train'][indices]
        sess.run(train_step, feed_dict={images_placeholder: images_batch,
                                        labels_placeholder: labels_batch})
        if i % 200 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                images_placeholder: images_batch, labels_placeholder: labels_batch})
            print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))
            test_accuracy = sess.run(accuracy, feed_dict={
                images_placeholder: data_sets['images_test'],
                labels_placeholder: data_sets['labels_test']})
            print('Test accuracy {:g}'.format(test_accuracy))
            if best_acc < test_accuracy:
                best_acc = test_accuracy
            print('Best accuracy {:g}'.format(best_acc))

    print('%d round training over' % (round_num + 1))
    print('time: %d ----> iter: %d ----> best_accuracy: %.4f' %
          (time.time() - start_time, local_epoch_num, best_acc))
    print('')
    sys.stdout.flush()

    # preparing update message, delta_model_paras is a list of numpy arrays
    new_model_paras = sess.run(tf.trainable_variables())
    delta_model_paras = [np.zeros(weights.shape) for weights in new_model_paras]
    for index in range(len(new_model_paras)):
        delta_model_paras[index] = new_model_paras[index] - old_model_paras[index]
    send_dict = {'model_paras': delta_model_paras}

    # update learning rate
    learning_rate *= decay_rate

    # connect to ps
    client_socket = communication.start_socket_client()
    # send updates
    client_socket.settimeout(300)
    while True:
        # communication.send_np_array(send_message, client_socket)
        send_message = pickle.dump(send_dict)
        communication.send_message(send_message, client_socket)
        print('Sent trained weights')
        sys.stdout.flush()
        break
    print("Client trains over in round %d  and takes %f second\n" % (round_num + 1, time.time() - start_time))
    print('-----------------------------------------------------------------')
    print('')
    print('')
    sys.stdout.flush()
    sess.close()

print('finished!')
sys.stdout.flush()