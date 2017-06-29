import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# load data
X_tr = np.load('train_im.npy')
y_tr = np.load('train_lb.npy')
X_te = np.load('test_im.npy')
y_te = np.load('test_lb.npy')

# define batch fcn
def batch(images,labels,index,batch_size):
	nsamples = images.shape[0]
	sub_start = index % nsamples
	sub_end = (index + batch_size) % nsamples
	if sub_end < sub_start:
		sub = range(sub_start,nsamples)	
		sub2 = range(0,sub_end)
		sub.extend(sub2)
	else:
		sub = range(sub_start,sub_end)
	return images[sub,:],labels[sub]

# NN fcn define
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Placeholders
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None,1])


# --- NN build -- # 

# 1st fc layer
W_fc1 = weight_variable([784,1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(x,W_fc1) + b_fc1)
# 2nd fc layer
W_fc2 = weight_variable([1024,1024])
b_fc2 = bias_variable([1024])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1,W_fc2) + b_fc2)
# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob)
# Readout layer
W_fc3 = weight_variable([1024,1])
b_fc3 = bias_variable([1])
y_conv = tf.matmul(h_fc2_drop,W_fc3) + b_fc3

# -- NN build -- #
cost = tf.reduce_sum(tf.abs(y_-y_conv))
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

# -- verify -- # 
plt.figure(figsize=[8,8])
tr_size_unit = y_tr.shape[0]/5
te_size_unit = y_te.shape[0]/5
for i in range(1,5):
	plt.subplot(2,4,i)
	plt.imshow(X_tr[i*tr_size_unit,:].reshape(28,28))
	plt.title("eval %g, real %g" %(int(y_conv.eval(feed_dict={x:np.array([X_tr[i*tr_size_unit,:]]),keep_prob:1.0})),int(y_tr[i*tr_size_unit])))
for i in range(1,5):
	plt.subplot(2,4,i+4)
	plt.imshow(X_te[i*te_size_unit,:].reshape(28,28))
	plt.title("eval %g, real %g" %(int(y_conv.eval(feed_dict={x:np.array([X_tr[i*tr_size_unit,:]]),keep_prob:1.0})),int(y_te[i*te_size_unit])))
plt.show()

#plt.plot(accuracy_table)
#plt.title('accuracy plot')
#plt.show()
