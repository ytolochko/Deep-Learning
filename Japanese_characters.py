# Here I train convolutional neural network classifier to recognize Japanese characters (Image Classification)

import h5py 
import numpy as np
import tensorflow as tf 
import os
import pandas as pd
import matplotlib.pyplot as plt

tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# load data 
def load_sample_dataset(filepath):
    

    with  h5py.File(filepath,'r') as hf:
        train_x = np.array(hf.get('train_x'))
        train_y = np.array(hf.get('train_y'))
        test_x = np.array(hf.get('test_x'))
        test_y = np.array(hf.get('test_y'))
    return train_x, train_y,test_x,test_y


X_train, y_train, X_test, y_test = load_sample_dataset("pmjt_sample_20161116/train_test_file_list.h5")

#normalzing images 

# normalzing training data 
for i in range(len(X_train)):
    X_train[i] = (X_train[i] - X_train[i].mean()) / X_train[i].std()

# normalzing test data 
for i in range(len(X_test)):
    X_test[i] = (X_test[i] - X_test[i].mean()) / X_test[i].std()

# reshaping x training values 
X_train = X_train.reshape(19909, 784)

# reshaping x test values 
X_test = X_test.reshape(3514, 784) 

# one hot vector for y test 
y_test = np.eye(10)[y_test]
y_test = y_test.reshape(3514, 10)

# one hot vector for y training 
y_train = np.eye(10)[y_train]
y_train = y_train.reshape(19909, 10)


def weight_variable(shape, name, train = True): 
    initial = tf.truncated_normal(shape, stddev=0.1)
    if train:
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(initial, name=name, trainable = False)


def bias_variable(shape, name, train = True): 
    initial = tf.constant(0.1, shape=shape)
    if train:
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(initial, name=name, trainable = False)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# this time alpha is not a constant but a variable, with intial value set to that which we determined to be most effcient in the previous subtask
alpha = tf.constant(0, dtype=tf.float32)

W_conv1 = weight_variable([5, 5, 1, 32], name = 'W_conv1')
b_conv1 = bias_variable([32], name = 'b_conv1')

x_image = tf.reshape(x, [-1, 28, 28, 1])

# create a boolean flag in order to differentiate between train and test when using batch normalization
training = tf.placeholder(tf.bool, name = 'training')

x_image = tf.reshape(x, [-1, 28, 28, 1])

# First layer
W_conv1 = weight_variable([5, 5, 1, 32], name = 'W_conv1')
b_conv1 = bias_variable([32], name = 'b_conv1')


h_conv1 = conv2d(x_image, W_conv1) + b_conv1

# add batch normalization
h_batch_norm1 = tf.contrib.layers.batch_norm(h_conv1, center = True, scale = True, is_training = training)

# add leaky ReLU. Formula is output = max(pixel*alpha, pixel). If pixel > 0, output = pixel. If pixel <=0, output = pixel * alpha (since alpha < 1).
# thus, it corresponds to the analytical formula of a leaky ReLU.
# the opertaion tf.maximum is element-wise - just what we need, thus we can feed in the whole matrix into tf.maximum

h_relu1 = tf.maximum(h_batch_norm1 * alpha, h_batch_norm1)
#h_relu1 = tf.nn.relu(h_batch_norm1)
h_pool1 = max_pool_2x2(h_relu1)

# Second layer

W_conv2 = weight_variable([5, 5, 32, 64], name = 'W_conv2')
b_conv2 = bias_variable([64], name = 'b_conv2')

h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
# add batch normalization.
h_batch_norm2 = tf.contrib.layers.batch_norm(h_conv2, center = True, scale = True, is_training = training)
# add leaky ReLU.
h_relu2 = tf.maximum(h_batch_norm2 * alpha, h_batch_norm2)
#h_relu2 = tf.nn.relu(h_batch_norm2)
h_pool2 = max_pool_2x2(h_relu2)

# Fully connected layer 1

W_fc1 = weight_variable([7 * 7 * 64, 1024], name = 'W_fc1')
b_fc1 = bias_variable([1024], name = 'b_fc1')

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# calculate the first fully connected layer
h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
# add batch normalization.
h_batch_norm3 = tf.contrib.layers.batch_norm(h_fc1, center = True, scale = True, is_training = training)
# add leaky ReLU, save in the same variable.
h_fc1 = tf.maximum(h_batch_norm3 * alpha, h_batch_norm3)  
#h_fc1 = tf.nn.relu(h_batch_norm3)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully connected layer 2 - last layer
W_fc2 = weight_variable([1024, 10], name = 'W_cf2')
b_fc2 = bias_variable([10], name = 'b_fc2')

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))


# Ensures that we execute the update_ops before performing the train_step
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# data structure for accuracies
accuracies_with_batch_normalization = {}
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1500):
        batch = next_batch(50, X_train, y_train)
        # record every 10 iterations
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0, training: 1})
            #print('step %d, training accuracy %g' % (i, train_accuracy))
            accuracies_with_batch_normalization[i/10] = train_accuracy
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, training: 1})
       
    print('test accuracy %g' % accuracy.eval(feed_dict={
      x: X_test, y_: y_test, keep_prob: 1.0, training: 0}))
    
    print('alpha =', sess.run(alpha))

# feed data into a pandas dataframe and plot it.
# Requires both accuracies (computed in previous file) and accuracies_with_batch_normalization (computed in this file) in memory

df = pd.DataFrame.from_dict(accuracies_with_batch_normalization).transpose()
df.columns = ['with_BN']
df.plot(figsize = (15,6))
plt.show()
