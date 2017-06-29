import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# load data
X_tr = np.load('train_im.npy')
y_tr = np.load('train_lb.npy')
X_te = np.load('test_im.npy')
y_te = np.load('test_lb.npy')c

# define batch fcn
def batch(images,labels,index,batch_size):
	nsamples = images.shape[0]
	sub_start = index % nsamples
	sub_end = (index + batch_size) % nsamples
	if sub_end < sub_start:
		subset = range(sub_start,nsamples)	
		subset2 = range(0,sub_end)
		subset.extend(subset2)
	else:
		subset = range(sub_start,sub_end)
	return images[subset,:],labels[subset,:]

# NN fcn define
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Placeholders
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None,1])


# --- ConvNet build start---
# 1st layer
W_conv1 = weight_variable([7,7,1,32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x,[-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
# 2nd layer
W_conv2 = weight_variable([7,7,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_conv1,W_conv2) + b_conv2) # the image is 16*16
# maxpool
h_pool2 = max_pool_2x2(h_conv2) # the images become 8*8
# FC layer
W_fc1 = weight_variable([8*8*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
# Readout layer
W_fc2 = weight_variable([1024,1])
b_fc2 = bias_variable([1])
y_conv = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
# -- ConvNet build end -- #

cost = tf.nn.l2_loss(y_-y_conv)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
batch_size = 50
accuracy = tf.reduce_mean(tf.cast(tf.less_equal(tf.abs(y_-y_conv),10.0),tf.float32))
sess.run(tf.global_variables_initializer())

accuracy_table = np.zeros([200,1])
for i in range(20000):
	x_batch,y_batch = batch(X_tr,y_tr,i,50)
	if i%100 == 0:
		cost_value = cost.eval(feed_dict={x:x_batch,y_:y_batch,keep_prob:1.0})
		train_accuracy = accuracy.eval(feed_dict={x:x_batch,y_:y_batch,keep_prob:1.0})
		accuracy_table[i/100] = train_accuracy
		print("step %d, training accuracy %g, cost is %g" %(i, train_accuracy,cost_value))
	train_step.run(feed_dict={x: x_batch, y_:y_batch, keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={x: X_te, y_: y_te, keep_prob: 1.0}))
plt.plot(accuracy_table)
plt.title('accuracy plot')
plt.show()
