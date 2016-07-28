# Example tool for exporting the Inception model for serving
#
# Download a model checkpoint file from
# http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz
#
# NOTE: This tool has to be built and ran thru bazel


import time
import sys

import tensorflow as tf
from tensorflow_serving.session_bundle import exporter
from inception import inception_model

NUM_CLASSES_TO_RETURN = 10


def convert_external_inputs(external_x):
    # transform the external input to the input format required on inference
    # convert the image string to a pixels tensor with values in the range 0,1
    image = tf.image.convert_image_dtype(tf.image.decode_jpeg(external_x, channels=3), tf.float32)
    # resize the image to the model expected width and height
    images = tf.image.resize_bilinear(tf.expand_dims(image, 0), [299, 299])
    # Convert the pixels to the range -1,1 required by the model
    images = tf.mul(tf.sub(images, 0.5), 2)
    return images


def inference(images):
    logits, _ = inception_model.inference(images, 1001)
    return logits



external_x = tf.placeholder(tf.string)
x = convert_external_inputs(external_x)
y = inference(x)

saver = tf.train.Saver()

with tf.Session() as sess:
    # Restore variables from training checkpoints.
    ckpt = tf.train.get_checkpoint_state(sys.argv[1])
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, sys.argv[1] + "/" + ckpt.model_checkpoint_path)
    else:
        print("Checkpoint file not found")
        raise SystemExit

    scores, class_ids = tf.nn.top_k(y, NUM_CLASSES_TO_RETURN)

    # for simplification we will just return the class ids, we should return the names instead
    classes = tf.contrib.lookup.index_to_string(tf.to_int64(class_ids),
        mapping=tf.constant([str(i) for i in range(1001)]))

    model_exporter = exporter.Exporter(saver)
    signature = exporter.classification_signature(
        input_tensor=external_x, classes_tensor=classes, scores_tensor=scores)
    model_exporter.init(default_graph_signature=signature, init_op=tf.initialize_all_tables())
    model_exporter.export(sys.argv[1] + "/export", tf.constant(time.time()), sess)

