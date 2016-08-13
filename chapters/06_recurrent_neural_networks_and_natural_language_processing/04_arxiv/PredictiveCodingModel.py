import tensorflow as tf
from helpers import lazy_property


class PredictiveCodingModel:

    def __init__(self, params, sequence, initial=None):
        self.params = params
        self.sequence = sequence
        self.initial = initial
        self.prediction
        self.state
        self.cost
        self.error
        self.logprob
        self.optimize

    @lazy_property
    def data(self):
        max_length = int(self.sequence.get_shape()[1])
        return tf.slice(self.sequence, (0, 0, 0), (-1, max_length - 1, -1))

    @lazy_property
    def target(self):
        return tf.slice(self.sequence, (0, 1, 0), (-1, -1, -1))

    @lazy_property
    def mask(self):
        return tf.reduce_max(tf.abs(self.target), reduction_indices=2)

    @lazy_property
    def length(self):
        return tf.reduce_sum(self.mask, reduction_indices=1)

    @lazy_property
    def prediction(self):
        prediction, _ = self.forward
        return prediction

    @lazy_property
    def state(self):
        _, state = self.forward
        return state

    @lazy_property
    def forward(self):
        cell = self.params.rnn_cell(self.params.rnn_hidden)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.params.rnn_layers)
        hidden, state = tf.nn.dynamic_rnn(
            inputs=self.data,
            cell=cell,
            dtype=tf.float32,
            initial_state=self.initial,
            sequence_length=self.length)
        vocabulary_size = int(self.target.get_shape()[2])
        prediction = self._shared_softmax(hidden, vocabulary_size)
        return prediction, state

    @lazy_property
    def cost(self):
        prediction = tf.clip_by_value(self.prediction, 1e-10, 1.0)
        cost = self.target * tf.log(prediction)
        cost = -tf.reduce_sum(cost, reduction_indices=2)
        return self._average(cost)

    @lazy_property
    def error(self):
        error = tf.not_equal(
            tf.argmax(self.prediction, 2), tf.argmax(self.target, 2))
        error = tf.cast(error, tf.float32)
        return self._average(error)

    @lazy_property
    def logprob(self):
        logprob = tf.mul(self.prediction, self.target)
        logprob = tf.reduce_max(logprob, reduction_indices=2)
        logprob = tf.log(tf.clip_by_value(logprob, 1e-10, 1.0)) / tf.log(2.0)
        return self._average(logprob)

    @lazy_property
    def optimize(self):
        gradient = self.params.optimizer.compute_gradients(self.cost)
        if self.params.gradient_clipping:
            limit = self.params.gradient_clipping
            gradient = [
                (tf.clip_by_value(g, -limit, limit), v)
                if g is not None else (None, v)
                for g, v in gradient]
        optimize = self.params.optimizer.apply_gradients(gradient)
        return optimize

    def _average(self, data):
        data *= self.mask
        length = tf.reduce_sum(self.length, 0)
        data = tf.reduce_sum(data, reduction_indices=1) / length
        data = tf.reduce_mean(data)
        return data

    def _shared_softmax(self, data, out_size):
        max_length = int(data.get_shape()[1])
        in_size = int(data.get_shape()[2])
        weight = tf.Variable(tf.truncated_normal(
            [in_size, out_size], stddev=0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[out_size]))
        # Flatten to apply same weights to all time steps.
        flat = tf.reshape(data, [-1, in_size])
        output = tf.nn.softmax(tf.matmul(flat, weight) + bias)
        output = tf.reshape(output, [-1, max_length, out_size])
        return output