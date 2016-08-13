import tensorflow as tf

from helpers import AttrDict

def get_params():
    checkpoint_dir = './arxiv-predictive-coding'
    max_length = 50
    sampling_temperature = 0.7
    rnn_cell = tf.nn.rnn_cell.GRUCell
    rnn_hidden = 200
    rnn_layers = 2
    learning_rate = 0.002
    optimizer = tf.train.AdamOptimizer(0.002)
    gradient_clipping = 5
    batch_size = 100
    epochs = 20
    epoch_size = 200
    return AttrDict(**locals())
