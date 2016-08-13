import tensorflow as tf

from helpers import AttrDict

from Embedding import Embedding
from ImdbMovieReviews import ImdbMovieReviews
from preprocess_batched import preprocess_batched
from SequenceClassificationModel import SequenceClassificationModel

IMDB_DOWNLOAD_DIR = './imdb'
WIKI_VOCAB_DIR = '../01_wikipedia/wikipedia'
WIKI_EMBED_DIR = '../01_wikipedia/wikipedia'

params = AttrDict(
    rnn_cell=tf.nn.rnn_cell.GRUCell,
    rnn_hidden=300,
    optimizer=tf.train.RMSPropOptimizer(0.002),
    batch_size=20,
)

reviews = ImdbMovieReviews(IMDB_DOWNLOAD_DIR)
length = max(len(x[0]) for x in reviews)

embedding = Embedding(
    WIKI_VOCAB_DIR + '/vocabulary.bz2',
    WIKI_EMBED_DIR + '/embeddings.npy', length)
batches = preprocess_batched(reviews, length, embedding, params.batch_size)

data = tf.placeholder(tf.float32, [None, length, embedding.dimensions])
target = tf.placeholder(tf.float32, [None, 2])
model = SequenceClassificationModel(data, target, params)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
for index, batch in enumerate(batches):
    feed = {data: batch[0], target: batch[1]}
    error, _ = sess.run([model.error, model.optimize], feed)
    print('{}: {:3.1f}%'.format(index + 1, 100 * error))
