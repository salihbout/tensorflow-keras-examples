#Import tensorflow and other libraries.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Generate random data : Rand :  uniform distribution over [0, 1).
number_of_examples = 100
x_data = np.random.rand(number_of_examples).astype(np.float32)
b = 0.4 + np.random.normal(scale=0.01, size=len(x_data))
y_data = x_data*0.1 +  b

#Visualizing the generated data
plt.plot(x_data,y_data, '.')
plt.show()


#Creating Tensorflow's variable
W = tf.Variable(tf.random_uniform([1],0.0,0.1))
B = tf.Variable(tf.zeros([1]))
Y = W*x_data + B

#build the training model
loss = tf.reduce_mean(tf.square(y_data - Y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()

#print the graph we build
print(tf.get_default_graph().as_graph_def())

ses = tf.Session()
ses.run(init)

y_initial = ses.run(Y)


#training the model :
number_iteration = 100

for step in range(number_iteration) :
    ses.run(train)
    if step % 10 == 0 :
        print('Iteration : ', step)
        best_weight, best_bais = ses.run([W, B])
        print("Weight : , Bais : ", best_weight, best_bais)
        print('Loss(Squared error) : ',ses.run(loss))



print("-----------------------------------")
print("Best W : ", best_weight[0])
print("Best B : ", best_bais[0])

#Visualizing the generated data

x_data = np.random.rand(number_of_examples).astype(np.float32)
b = 0.4 + np.random.normal(scale=0.01, size=len(x_data))
y_data = x_data*0.1 +  b

#Visualizing the generated data
plt.plot(x_data,y_data, '.')



Ys = best_weight*x_data + best_bais
plt.plot(x_data, Ys, '-', c="red")

# x_data = np.random.rand(100).astype(np.float32)
# b = 0.4 + np.random.normal(scale=0.01, size=len(x_data))
# y_data = x_data*0.1 +  b
# plt.plot(x_data,y_data, '.')
plt.show()
