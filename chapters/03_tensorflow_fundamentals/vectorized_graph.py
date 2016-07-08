import tensorflow as tf
import numpy as np

# Explicitly create a Graph object
graph = tf.Graph()

with graph.as_default():
    
    with tf.name_scope("variables"):
        # Variable to keep track of how many times the graph has been run
        global_step = tf.Variable(0, dtype=tf.int32, name="global_step")
        
        # Increments the above `global_step` Variable, should be run whenever the graph is run
        increment_step = global_step.assign_add(1)
        
        # Variable that keeps track of previous output value:
        previous_value = tf.Variable(0.0, dtype=tf.float32, name="previous_value")
    
    # Primary transformation Operations
    with tf.name_scope("exercise_transformation"):
        
        # Separate input layer
        with tf.name_scope("input"):
            # Create input placeholder- takes in a Vector 
            a = tf.placeholder(tf.float32, shape=[None], name="input_placeholder_a")
    
        # Separate middle layer
        with tf.name_scope("intermediate_layer"):
            b = tf.reduce_prod(a, name="product_b")
            c = tf.reduce_sum(a, name="sum_c")
        
        # Separate output layer
        with tf.name_scope("output"):
            d = tf.add(b, c, name="add_d")
            output = tf.sub(d, previous_value, name="output")
            update_prev = previous_value.assign(output)
    
    # Summary Operations
    with tf.name_scope("summaries"):
        tf.scalar_summary(b'output', output, name="output_summary")  # Creates summary for output node
        tf.scalar_summary(b'product of inputs', b, name="prod_summary")
        tf.scalar_summary(b'sum of inputs', c, name="sum_summary")
    
    # Global Variables and Operations
    with tf.name_scope("global_ops"):
        # Initialization Op
        init = tf.initialize_all_variables()
        # Collect all summary Ops in graph
        merged_summaries = tf.merge_all_summaries()

# Start a Session, using the explicitly created Graph
sess = tf.Session(graph=graph)

# Open a SummaryWriter to save summaries
writer = tf.train.SummaryWriter('./improved_graph', graph)

# Initialize Variables
sess.run(init)

def run_graph(input_tensor):
    """
    Helper function; runs the graph with given input tensor and saves summaries
    """
    feed_dict = {a: input_tensor}
    output, summary, step = sess.run([update_prev, merged_summaries, increment_step], feed_dict=feed_dict)
    writer.add_summary(summary, global_step=step)


# Run the graph with various inputs
run_graph([2,8])
run_graph([3,1,3,3])
run_graph([8])
run_graph([1,2,3])
run_graph([11,4])
run_graph([4,1])
run_graph([7,3,1])
run_graph([6,3])
run_graph([0,2])
run_graph([4,5,6])

# Writes the summaries to disk
writer.flush()

# Flushes the summaries to disk and closes the SummaryWriter
writer.close()

# Close the session
sess.close()

# To start TensorBoard after running this file, execute the following command:
# $ tensorboard --logdir='./improved_graph'