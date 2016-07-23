# Errata - Chapter 4: Machine Learning Basics

**Full book page no.

* **p.111** "If you encode the symbols assuming any other probability for the q_i than the real p_i need more bits for encoding each symbol.". Should read as "...than the real p_i, you need more..." - _Found 7/22/2016_

* **p.115** "You can proof that if...". Should read as "You can prove that if..." - _Found 7/22/2016_

* **p.115** "b = tf.Variable(tf.zeros([3], name="bias"))". Should read as "b = tf.Variable(tf.zeros([3]), name="bias")" - _Found 7/22/2016_

* **p.117** "predicted = tf.cast(tf.arg_max(inference(X), 1), tf.int32)" and then "We use the tf.argmax function to choose...". The code part should read "tf.argmax" for consistency with the rest of the code so far, as well as it being visible in the TensorFlow documentation - _Found 7/22/2016_

* **p.120** "We can’t find a single stright line that would split the chart..." Should say "straight" - _Found 7/22/2016_

* **p.121** "Then you can leave all of the equal outputs grouped toghether in a single area." Should say "together" - _Found 7/22/2016_

* **p.128** "Tensorflow includes the method tf.gradients to simbolically computate the gradients..." Should say "symbolically compute" - _Found 7/22/2016_

* **p.129** "loss = cross<sub>e</sub>ntropy L3, y_expected)" Should read as "cross_entropy" - _Found 7/22/2016_
