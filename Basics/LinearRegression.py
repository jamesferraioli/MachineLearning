#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function


# In[2]:


import tensorflow as tf


# In[11]:


# Basic constant operations
# The value returned by the constructor represents the output
# of the Constant op.
a = tf.constant(2)


# In[9]:


b = tf.constant(3)


# In[12]:


# Launch the default graph.
with tf.Session() as sess:
    print("a=2, b=3")
    print("Addition with constants: %i" % sess.run(a+b))
    print("Multiplication with constants: %i" % sess.run(a*b))


# In[13]:


# Basic Operations with variable as graph input
# The value returned by the constructor represents the output
# of the Variable op. (define as input when running session)
# tf Graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)


# In[14]:


# Define some operations
add = tf.add(a, b)
mul = tf.multiply(a, b)


# In[15]:


# Launch the default graph.
with tf.Session() as sess:
    # Run every operation with variable input
    print("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))


# In[16]:


# A linear regression learning algorithm example using TensorFlow library.


# In[17]:


import numpy


# In[18]:


import matplotlib.pyplot as plt


# In[19]:


rng = numpy.random


# In[20]:


#params
learning_rate = 0.01
training_epochs = 1000
display_step = 50


# In[34]:


# training data
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]


# In[22]:


#tf Graph input
X = tf.placeholder("float")
Y = tf.placeholder("float")


# In[23]:


# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")


# In[24]:


#construct a linear model


# In[25]:


pred = tf.add(tf.multiply(X,W), b)


# In[26]:


#mean squared error


# In[27]:


cost = tf.reduce_sum(tf.pow(pred-Y,2))/(2*n_samples)


# In[28]:


#gradient descent


# In[29]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# In[30]:


#initialize the variabls (assign default value)


# In[31]:


init = tf.global_variables_initializer()


# In[32]:


#start training


# In[36]:


with tf.Session() as sess:
    #run in the initializer
    sess.run(init)
    
    #fit all training data
    for epoch in range(training_epochs):
        for(x,y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
            
        #display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),                  "W=", sess.run(W), "b=", sess.run(b))
            
    print("optimization finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
    
    #graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()


# In[ ]:




