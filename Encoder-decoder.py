# Here I build an encoder-decoder network which translates the subset of Cyrillic 
# handwritten glyphs for which there is a unique correspondence into the matching Latin sounds, expressed
# as handwritten glyphs again

from __future__ import division, print_function, absolute_import
import tensorflow.contrib.layers as lays
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform

# Load Raw Data
cyr = np.load('data\\X_cyr.npy')
lat = np.load('data\\X_lat.npy')
cyr_labels = np.load('data\\labels_cyr.npy')
lat_labels = np.load('data\\labels_lat.npy')

# Create Training Data
cyr_list = []
lat_list = []

for ci, i in enumerate(cyr_labels):
    counter = 0 
    for cj, j in enumerate(lat_labels):
        if i == j:
            cyr_list.append(cyr[:, :, ci])
            lat_list.append(lat[:, :, cj])
            counter += 1 
        if counter > 4: 
            break 
    
            
cyr = np.stack(cyr_list, axis=0)
lat = np.stack(lat_list, axis=0)

def resize_batch(imgs):
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
    return resized_imgs

cyr = cyr.reshape((-1, 28, 28, 1))
lat = lat.reshape((-1, 28, 28, 1))
cyr = resize_batch(cyr)
lat = resize_batch(lat)

# normalize the images
cyr = (cyr - np.mean(cyr))/np.std(cyr)
lat = (lat - np.mean(lat))/np.std(lat)

# divide into train (80%) and test (20%)
train_size = int(cyr.shape[0]*0.8)
test_size = cyr.shape[0] - train_size

# create ids with the size of the input
ids = np.arange(0,cyr.shape[0])
# randomly shuffle the ids
np.random.shuffle(ids)
# select the ids into two categories - train and test
train_ids = ids[:train_size]
test_ids = ids[train_size : train_size + test_size]

# cyrillic
cyr_train = cyr[train_ids]
cyr_test = cyr[test_ids]

# latin
lat_train = lat[train_ids]
lat_test = lat[test_ids]


batch_size = 100  # Number of samples in each batch
lr = 0.001        # Learning rate

# andreas

def encoder(inputs):
    # encoder
    net = lays.conv2d(inputs, 28, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 8, [5, 5], stride=4, padding='SAME')
    return net

def decoder(inputs):
    net2 = lays.conv2d_transpose(inputs, 16, [5, 5], stride=4, padding='SAME')
    net2 = lays.conv2d_transpose(net2, 28, [5, 5], stride=2, padding='SAME')
    net2 = lays.conv2d_transpose(net2, 1, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.relu)
    return net2


x = tf.placeholder(tf.float32, (None, 32, 32, 1)) 
#x_image = tf.reshape(x, [-1, 32, 32, 1])


y_true = tf.placeholder(tf.float32, (None, 32, 32, 1)) 
#y_image = tf.reshape(y_true, [-1, 32, 32, 1])


z = encoder(x)
y = decoder(z)  

# calculate the loss and optimize the network
loss = tf.reduce_mean(tf.square(y_true - y))  
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


for i in range(10000):
    batch_xs, batch_ys = next_batch(100, cyr_train, lat_train)
    sess.run(train_step, feed_dict={x: batch_xs, y_true: batch_ys})
    if i % 1000 == 0:
            train_loss = loss.eval(feed_dict={x: batch_xs, y_true: batch_ys})
            print('step %d, training accuracy %g' % (i, train_loss))

test_loss = loss.eval(feed_dict = {x: cyr_test, y_true: lat_test})
print('test loss = ', test_loss)

# we want to visualize the results, just evaluate y (recovered image) on some inputs from test dataset            
recovered_image = sess.run(y, feed_dict = {x : cyr_test[0:50]})

plt.imshow(cyr_test[5, :, :, 0])
plt.show()


plt.imshow(lat_test[5, :, :, 0])
plt.show()


plt.imshow(recovered_image[5,:,:,0])
plt.show()
  
