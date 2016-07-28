# Errata - Chapter 3: TensorFlow Fundamentals

This chapter was included in a sample release of the book, so we include the page number for the sample in parentheses:

**Full book page no. (sample chapters page no.)

* **p.57: (p.19)** "...the function automatically converts the scalar numbers 6 and 3 into `Tensor` objects...". The number "6" should be "5" - _Found 7/21/2016_

* **p.65 (p.27:**) "d = tf.add(c,d, name='add_d')". The `c,d` parameters should be `b,c` in this case. - _Found 7/21/2016_

* **p.70: (p.32)** "Yes, that means there are Ops that technically take in zero inputs and zero outputs.". This should read "take in zero inputs and return zero outputs" - _Found 7/21/2016_

* **p.72: (p.34)** "x ⇐ y tf.less_equal()" and "Returns the truth table of x ⇐ y, element-wise". '⇐' should read as <= to be consistent with the line for tf.greater_equal() below it - _Found 7/21/2016_

* **p.84: (p.46)** "...each name scopes will encapsulate its own Ops...". Should read as "...each name scope will encapsulate its own Ops..." - _Found 7/21/2016_

