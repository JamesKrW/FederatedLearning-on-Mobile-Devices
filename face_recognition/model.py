import numpy as np
import tensorflow as tf
from data_process import load_data
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
batch_size = 32
learning_rate = 0.001
epoch = 5000
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

saver = tf.train.Saver()
if not os.path.exists('./tmp/'):
    os.mkdir('./tmp/')

with tf.Session() as sess:
    if os.path.exists('./tmp/checkpoint'):
        saver.restore(sess, './tmp/model.ckpt')
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
    data_sets = load_data()
    best_acc = 0
    for i in range(epoch):
        indices = np.random.choice(data_sets['images_train'].shape[0], batch_size)
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
                save_path = saver.save(sess, './tmp/model.ckpt')
            print('Best accuracy {:g}'.format(best_acc))

