import numpy as np
import tensorflow as tf
from data_process import load_data
import os


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
batch_size = 32
learning_rate = 0.001
epoch = 100000
persons = 4


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = tf.keras.layers.Dense(64, tf.nn.leaky_relu, kernel_regularizer=tf.keras.regularizers.l2(0.05))
        self.drop_layer1 = tf.keras.layers.Dropout(0.2)
        self.FCN = tf.keras.layers.Dense(persons)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.drop_layer1(x)
        output = self.FCN(x)
        output_with_sm = tf.nn.softmax(output)
        return output, output_with_sm


if __name__ == '__main__':
    # placeholder
    input_place_holder = tf.placeholder(tf.float32, [None, 128], name='input')
    label_place_holder = tf.placeholder(tf.int32, [None], name='label')
    label_place_holder_2d = tf.one_hot(label_place_holder, persons)
    # model
    model = MyModel()
    output, output_with_sm = model(input_place_holder)
    # loss function
    bce = tf.keras.losses.CategoricalCrossentropy()
    loss = bce(label_place_holder_2d, output_with_sm)
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    # acc
    prediction_place_holder = tf.placeholder(tf.float64, [None], name='pred')
    acc, update_op = tf.metrics.accuracy(labels=label_place_holder, predictions=prediction_place_holder)
    # run
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # load data
    data_sets = load_data()
    with tf.Session() as sess:
        sess.run(init_op)
        saver = tf.train.Saver()
        best_val_acc = 0
        for itr in range(epoch):
            indices = np.random.choice(data_sets['images_train'].shape[0], batch_size)
            images_batch = data_sets['images_train'][indices]
            labels_batch = data_sets['labels_train'][indices]
            _, loss_ = sess.run([train_op, loss], {'input:0': images_batch, 'label:0': labels_batch})
            #print("iter {}, training set loss: {:.4}".format(itr, loss_))
            if itr % 1000 == 0:
                prediction = sess.run(output_with_sm, {'input:0': data_sets['images_test']})
                prediction = prediction[:, 1]
                acc_value = sess.run(update_op,
                                     feed_dict={prediction_place_holder: prediction,
                                                label_place_holder: data_sets['labels_test']})
                print("itr={}, acc={} ".format(itr, acc_value))
                if acc_value > best_val_acc:
                    best_val_acc = acc_value
                    saver.save(sess, './weights/model')
                print("best acc: ", best_val_acc)
