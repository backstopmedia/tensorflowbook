import tensorflow as tf
import numpy as np

from helpers import overwrite_graph
from Preprocessing import Preprocessing
from PredictiveCodingModel import PredictiveCodingModel

class Sampling:

    @overwrite_graph
    def __init__(self, params):
        self.params = params
        self.prep = Preprocessing([], 2, self.params.batch_size)
        self.sequence = tf.placeholder(
            tf.float32, [1, 2, len(self.prep.VOCABULARY)])
        self.state = tf.placeholder(
            tf.float32, [1, self.params.rnn_hidden * self.params.rnn_layers])
        self.model = PredictiveCodingModel(
            self.params, self.sequence, self.state)
        self.sess = tf.Session()
        checkpoint = tf.train.get_checkpoint_state(self.params.checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            tf.train.Saver().restore(
                self.sess, checkpoint.model_checkpoint_path)
        else:
            print('Sampling from untrained model.')
        print('Sampling temperature', self.params.sampling_temperature)

    def __call__(self, seed, length=100):
        text = seed
        state = np.zeros((1, self.params.rnn_hidden * self.params.rnn_layers))
        for _ in range(length):
            feed = {self.state: state}
            feed[self.sequence] = self.prep([text[-1] + '?'])
            prediction, state = self.sess.run(
                [self.model.prediction, self.model.state], feed)
            text += self._sample(prediction[0, 0])
        return text

    def _sample(self, dist):
        dist = np.log(dist) / self.params.sampling_temperature
        dist = np.exp(dist) / np.exp(dist).sum()
        choice = np.random.choice(len(dist), p=dist)
        choice = self.prep.VOCABULARY[choice]
        return choice