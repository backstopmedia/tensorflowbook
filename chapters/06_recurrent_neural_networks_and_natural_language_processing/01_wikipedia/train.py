import collections
import tensorflow as tf
import numpy as np

from batched import batched
from EmbeddingModel import EmbeddingModel
from skipgrams import skipgrams
from Wikipedia import Wikipedia

from helpers import AttrDict

WIKI_DOWNLOAD_DIR = './wikipedia'

params = AttrDict(
    vocabulary_size=10000,
    max_context=10,
    embedding_size=200,
    contrastive_examples=100,
    learning_rate=0.5,
    momentum=0.5,
    batch_size=1000,
)

data = tf.placeholder(tf.int32, [None])
target = tf.placeholder(tf.int32, [None])
model = EmbeddingModel(data, target, params)

corpus = Wikipedia(
    'https://dumps.wikimedia.org/enwiki/20160501/'
    'enwiki-20160501-pages-meta-current1.xml-p000000010p000030303.bz2',
    WIKI_DOWNLOAD_DIR,
    params.vocabulary_size)
examples = skipgrams(corpus, params.max_context)
batches = batched(examples, params.batch_size)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
average = collections.deque(maxlen=100)
for index, batch in enumerate(batches):
    feed_dict = {data: batch[0], target: batch[1]}
    cost, _ = sess.run([model.cost, model.optimize], feed_dict)
    average.append(cost)
    print('{}: {:5.1f}'.format(index + 1, sum(average) / len(average)))
    if index > 100000:
        break

embeddings = sess.run(model.embeddings)
np.save(WIKI_DOWNLOAD_DIR + '/embeddings.npy', embeddings)
