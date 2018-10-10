import tensorflow as tf
from tensorflow import keras
from math import floor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
tf.reset_default_graph()

# Parameters:
epochs = 5
learning_rate = 0.001
batch_size = 128

n_input = 28
n_classes = 10


# import data
X_test = pd.read_csv('test.csv')
train_df = pd.read_csv('train.csv')

X_test.shape
train_df.shape

# split the training data into training and vaidation
train_df, val_df = train_test_split(train_df)

train_df.shape
val_df.shape

# seperate out the X and Y data frames
X_train = train_df.drop(['label'], axis=1)
Y_train = train_df['label']

X_val = val_df.drop(['label'], axis=1)
Y_val = val_df['label']

# one hot encode
number_of_classes = 10
Y_train = keras.utils.to_categorical(Y_train, number_of_classes)
Y_val = keras.utils.to_categorical(Y_val, number_of_classes)

# Reshape to [samples][pixel_rows][pixel_cols][channels]
X_train = np.array(X_train).reshape(len(X_train), 28, 28, 1)
X_val = np.array(X_val).reshape(len(X_val), 28, 28, 1)
X_test = np.array(X_test).reshape(len(X_test), 28, 28, 1)


# look at shapes
X_train.shape
Y_train.shape

X_val.shape
Y_val.shape


# Scale the pixel data
X_train.max()
X_train, X_val, X_test = X_train / 255, X_val / 255, X_test / 255

X_train.max()


# Random rotations
datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=45)
datagen.fit(X_train)


# placeholders for input and labels:
x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(tf.float32, shape=[None, n_classes])


# create layer wrappers:

def conv2d(x, W, b, strides=1):
    x_1 = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x_2 = tf.add(x_1, b)
    x_3 = tf.nn.relu(x_2)
    return x_3


def maxpool2d(x, k=2):
    x_1 = tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[
                         1, k, k, 1], padding='SAME')
    return x_1


# Now define weights and bias dictionaries

weights = {
    'wc1': tf.get_variable('W0', shape=(3, 3, 1, 32)),
    'wc2': tf.get_variable('W1', shape=(3, 3, 32, 64)),
    'wc3': tf.get_variable('W2', shape=(3, 3, 64, 128)),
    'wd1': tf.get_variable('W3', shape=(4 * 4 * 128, 128)),
    'out': tf.get_variable('W6', shape=(128, n_classes))
}

biases = {
    'bc1': tf.get_variable('B0', shape=(32)),
    'bc2': tf.get_variable('B1', shape=(64)),
    'bc3': tf.get_variable('B2', shape=(128)),
    'bd1': tf.get_variable('B3', shape=(128)),
    'out': tf.get_variable('B4', shape=(n_classes))
}


# Now create the whole neural network
def conv_net(x, weights, biases):
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)

    fc1 = tf.reshape(conv3, [-1, weights['wd1'].shape.as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Now compute loss
pred = conv_net(x, weights, biases)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# evaluate accuracy:
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# output predictions
preds = tf.argmax(pred, 1)

# Initialize variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    valid_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(epochs):
        for batch in range(floor(len(X_train)/batch_size)):
            batch_x = X_train[batch*batch_size:min((batch+1)*batch_size,len(X_train))]
            batch_y = Y_train[batch*batch_size:min((batch+1)*batch_size,len(Y_train))]

            opt = sess.run(optimizer, feed_dict = {x: batch_x , y: batch_y})
            loss, acc = sess.run([cost,accuracy], feed_dict = {x: batch_x , y: batch_y})

        print('Iteration ' + str(i) + ", Loss= " + \
        "{:.6f}".format(loss) + ", Training Accuracy= " + \
        "{:.5f}".format(acc))

        # calculate accuracy for all training and validation:
        valid_acc = sess.run(accuracy, feed_dict = {x : X_val, y: Y_val})
        print('Validation Accuracy= '+ "{:.5f}".format(valid_acc))
        valid_accuracy.append(valid_acc)
    summary_writer.close()
    predictions = sess.run(preds, feed_dict = {x : X_test})
    output = pd.concat([pd.Series(range(X_test.shape[0])),
                        pd.to_numeric(pd.Series(predictions),downcast='integer')], axis = 1)

    output.columns = ['ImageId','Label']
    output.to_csv('output.csv',index=False)
