import dataset
import numpy as np
import matplotlib.pyplot as plt 

# load data
X,y = dataset.load_mnist()

size = y.size
X_n = np.zeros([size,784])
y_n = np.zeros([size,1])

# split to train and test
train_ratio = 0.4

# train set
train_num = int(train_ratio * size)
rand = np.random.random([train_num])*size
train_ind = rand.astype(int)
X_tr = X[train_ind,:]
y_tr = y[train_ind]

# test set
total = range(size)
test_ind = np.array(list(set(total) - set(train_ind)))
X_te = X[test_ind,:]
y_te = y[test_ind]

print "train size",X_tr.shape,y_tr.shape
print "test size",X_te.shape,y_te.shape

# --- Verify dataset ---
plt.figure(figsize=[8,8])
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
