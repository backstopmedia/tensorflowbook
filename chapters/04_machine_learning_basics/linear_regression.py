# Linear regression example in TF.

import tensorflow as tf

W = tf.Variable(tf.zeros([2, 1]), name="weights")
b = tf.Variable(0., name="bias")


def inference(X):
    return tf.matmul(X, W) + b


def loss(X, Y):
    Y_predicted = tf.transpose(inference(X)) # make it a row vector
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))


def inputs():
    # Data from http://people.sc.fsu.edu/~jburkardt/datasets/regression/x09.txt
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [63, 28], [72, 36], [79, 57], [75, 44], [27, 24], [89, 31], [65, 52], [57, 23], [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34], [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]
    blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395, 434, 220, 374, 308, 220, 311, 181, 274, 303, 244]

    return tf.to_float(weight_age), tf.to_float(blood_fat_content)


def train(total_loss):
    learning_rate = 0.000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):
    print sess.run(inference([[50., 20.]])) # ~ 303
    print sess.run(inference([[50., 70.]])) # ~ 256
    print sess.run(inference([[90., 20.]])) # ~ 303
    print sess.run(inference([[90., 70.]])) # ~ 256

# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:

    tf.initialize_all_variables().run()

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # actual training loop
    training_steps = 10000
    for step in range(training_steps):
        sess.run([train_op])
        if step % 1000 == 0:
            print "Epoch:", step, " loss: ", sess.run(total_loss)

    print "Final model W=", sess.run(W), "b=", sess.run(b)
    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    sess.close()


