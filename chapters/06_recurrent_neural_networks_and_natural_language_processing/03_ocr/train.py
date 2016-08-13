import random

import tensorflow as tf
import numpy as np

from helpers import AttrDict

from OcrDataset import OcrDataset
from SequenceLabellingModel import SequenceLabellingModel
from batched import batched

params = AttrDict(
    rnn_cell=tf.nn.rnn_cell.GRUCell,
    rnn_hidden=300,
    optimizer=tf.train.RMSPropOptimizer(0.002),
    gradient_clipping=5,
    batch_size=10,
    epochs=5,
    epoch_size=50
)

def get_dataset():
    dataset = OcrDataset('./ocr')
    # Flatten images into vectors.
    dataset.data = dataset.data.reshape(dataset.data.shape[:2] + (-1,))
    # One-hot encode targets.
    target = np.zeros(dataset.target.shape + (26,))
    for index, letter in np.ndenumerate(dataset.target):
        if letter:
            target[index][ord(letter) - ord('a')] = 1
    dataset.target = target
    # Shuffle order of examples.
    order = np.random.permutation(len(dataset.data))
    dataset.data = dataset.data[order]
    dataset.target = dataset.target[order]
    return dataset

# Split into training and test data.
dataset = get_dataset()
split = int(0.66 * len(dataset.data))
train_data, test_data = dataset.data[:split], dataset.data[split:]
train_target, test_target = dataset.target[:split], dataset.target[split:]

# Compute graph.
_, length, image_size = train_data.shape
num_classes = train_target.shape[2]
data = tf.placeholder(tf.float32, [None, length, image_size])
target = tf.placeholder(tf.float32, [None, length, num_classes])
model = SequenceLabellingModel(data, target, params)
batches = batched(train_data, train_target, params.batch_size)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
for index, batch in enumerate(batches):
    batch_data = batch[0]
    batch_target = batch[1]
    epoch = batch[2]
    if epoch >= params.epochs:
        break
    feed = {data: batch_data, target: batch_target}
    error, _ = sess.run([model.error, model.optimize], feed)
    print('{}: {:3.6f}%'.format(index + 1, 100 * error))

test_feed = {data: test_data, target: test_target}
test_error, _ = sess.run([model.error, model.optimize], test_feed)
print('Test error: {:3.6f}%'.format(100 * error))