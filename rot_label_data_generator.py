import dataset
import numpy as np
import matplotlib.pyplot as plt 
import cv2

# load data
X,y = dataset.load_mnist()

y_size = y.size
X_n = np.zeros([y_size,784])
y_n = np.zeros([y_size,1])

# pick up digits 2
index = 0
for i in range(y.size):
	if y[i] == 2:
		X_n[index,:] = X[i,:]
		y_n[index,:] = y[i]
		index += 1
X_n = np.delete(X_n,range(index,y_size+1),0)
y_n = np.delete(y_n,range(index,y_size+1),0)

cols = 28
rows = 28

num = X_n.shape[0]

X_r = np.zeros([num*12,784])
y_r = np.zeros([num*12,1])

# Make new dataset
for i in range(11):
	for j in range(num):
		variation = 10 * (np.random.random()-0.5)
		rotate_degrees = 30 * i + variation
		M = cv2.getRotationMatrix2D((cols/2,rows/2),rotate_degrees,1)
		x = X_n[j,:].reshape(28,28)
		dst = cv2.warpAffine(x,M,(cols,rows))
		X_r[i*num+j,:] = dst.reshape(1,784)
		y_r[i*num+j,:] = [rotate_degrees]

# split to train and test
train_ratio = 0.4
size = X_r.shape[0]

# train set
train_num = int(train_ratio * size)
rand = np.random.random([train_num])*size
train_ind = rand.astype(int)
X_tr = X_r[train_ind,:]
y_tr = y_r[train_ind,:]

# test set
total = range(size)
test_ind = np.array(list(set(total) - set(train_ind)))
X_te = X_r[test_ind,:]
y_te = y_r[test_ind,:]

# --- Verify dataset ---
plt.figure(figsize=[10,10])
tr_size_unit = y_tr.shape[0]/5
te_size_unit = y_te.shape[0]/5
for i in range(1,5):
	plt.subplot(2,4,i)
	plt.imshow(X_tr[i*tr_size_unit,:].reshape(28,28))
	plt.title(y_tr[i*tr_size_unit])
for i in range(1,5):
	plt.subplot(2,4,i+4)
	plt.imshow(X_te[i*te_size_unit,:].reshape(28,28))
	plt.title(y_te[i*te_size_unit])
plt.show()

# --- Save dataset ---
np.save('train_im',X_tr)
np.save('train_lb',y_tr)
np.save('test_im',X_te)
np.save('test_lb',y_te)

