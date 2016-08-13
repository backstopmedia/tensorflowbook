import tensorflow as tf

from helpers import lazy_property


class BidirectionalSequenceLabellingModel:

    def __init__(self, data, target, params):
        self.data = data
        self.target = target
        self.params = params
        self.prediction
        self.cost
        self.error
        self.optimize

    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        output = self._bidirectional_rnn(self.data, self.length)
        num_classes = int(self.target.get_shape()[2])
        prediction = self._shared_softmax(output, num_classes)
        return prediction

    def _bidirectional_rnn(self, data, length):
        length_64 = tf.cast(length, tf.int64)
        forward, _ = tf.nn.dynamic_rnn(
            cell=self.params.rnn_cell(self.params.rnn_hidden),
            inputs=data,
            dtype=tf.float32,
            sequence_length=length,
            scope='rnn-forward')
        backward, _ = tf.nn.dynamic_rnn(
            cell=self.params.rnn_cell(self.params.rnn_hidden),
            inputs=tf.reverse_sequence(data, length_64, seq_dim=1),
            dtype=tf.float32,
            sequence_length=self.length,
            scope='rnn-backward')
        backward = tf.reverse_sequence(backward, length_64, seq_dim=1)
        output = tf.concat(2, [forward, backward])
        return output

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

    @lazy_property
    def cost(self):
        # Compute cross entropy for each frame.
        cross_entropy = self.target * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        cross_entropy *= mask
        # Average over actual sequence lengths.
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
        mistakes = tf.cast(mistakes, tf.float32)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        mistakes *= mask
        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
        mistakes /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(mistakes)

    @lazy_property
    def optimize(self):
        gradient = self.params.optimizer.compute_gradients(self.cost)
        try:
            limit = self.params.gradient_clipping
            gradient = [
                (tf.clip_by_value(g, -limit, limit), v)
                if g is not None else (None, v)
                for g, v in gradient]
        except AttributeError:
            print('No gradient clipping parameter specified.')
        optimize = self.params.optimizer.apply_gradients(gradient)
        return optimize
