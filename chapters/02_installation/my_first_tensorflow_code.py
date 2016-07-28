import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

a = tf.random_normal([2,20])
sess = tf.Session()
out = sess.run(a)
x, y = out

plt.scatter(x, y)
plt.show()