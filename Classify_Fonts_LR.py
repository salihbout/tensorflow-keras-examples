### CLASSIFY FONTS USING A SIMPLE LINEAR REGRESSION : 35% accuracy

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x


#Set a seed for reproducibility
np.random.seed(0)

#Load the data
data = np.load('data_n_labels.npz')

#Preproccessing the data
train = data['arr_0'] / 255
labels = data['arr_1']

#visualizing some data
plt.ion()
# plt.figure( figsize=(6,6))
# f, plts = plt.subplots(5,sharex=True)
# c=91
#
# for i in range(5) :
#     plts[i].pcolor(train[c+i*558], cmap=plt.cm.gray_r)


# Transform labels to one hot ( one zero format )
def data_to_onhot(labels, numb_classes = 5) :
    outlabels = np.zeros((len(labels), numb_classes))
    for i, j in enumerate(labels):
        outlabels[i,j] = 1

    return outlabels


onehoted_data = data_to_onhot(labels)

#Splitting the data into training and validation
indices = np.random.permutation(train.shape[0])
validation_range = int(train.shape[0]*0.1)
test_idx , train_idx = indices[:validation_range],  indices[validation_range:]
test, train = train[test_idx,:], train[train_idx, :]

onehot_test , onehot_train = onehoted_data[test_idx,:], onehoted_data[train_idx, :]

#Hello Tensorflow !

session = tf.InteractiveSession()

#inputs
X = tf.placeholder("float", [None, 1296])
Y_ = tf.placeholder("float", [None, 5])

#Variables
W = tf.Variable(tf.zeros([1296, 5]))
b = tf.Variable(tf.zeros([5]))

#Initialize :
session.run(tf.initialize_all_variables())


#LR model
Y = tf.nn.softmax(tf.matmul(X,W) + b)

#Cost function & train !
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y ,labels=Y_))
train_step = tf.train.GradientDescentOptimizer(0.02).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#Let's train now !
num_epochs = 1000
train_acc = np.zeros(num_epochs//10)
test_acc = np.zeros(num_epochs//10)

for i in tqdm(range(num_epochs)) :
    if i % 10 ==0 :
        accu = accuracy.eval(feed_dict={
            X: train.reshape([-1, 1296]),
            Y_: onehot_train})

        train_acc[i//10] = accu

        accu = accuracy.eval(feed_dict={
            X: test.reshape([-1, 1296]),
            Y_: onehot_test})

        test_acc[i // 10] = accu

    train_step.run(feed_dict={
            X: train.reshape([-1, 1296]),
            Y_: onehot_train})

print(train_acc[-1])
print(test_acc[-1])

plt.figure(figsize=(6,6))
plt.plot(train_acc, 'bo')
plt.plot(test_acc, 'rx')






