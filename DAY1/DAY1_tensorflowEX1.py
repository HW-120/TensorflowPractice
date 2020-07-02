"""
try:
    # %tensorflow_version only exists in Colab.
    %tensorflow_version 1.5 이건데 왜 안되징
except Exception:
    pass
import tensorflow as tf
import numpy as np

# @tf.function    # 데코레이터 나중에 확실하게 되면 이걸 써줘
a = [[1.0, 2.0], [3.0, 4.0]]
b = np.array([[1.0, 2.0],
              [3.0, 4.0]])
c = tf.constant([[1.0, 2.0],[3.0, 4.0]])

print(type(a))
print(type(tf.convert_to_tensor(a, dtype=tf.float32)))

# Tensorflow 1.X 에서 하는 것
hello = tf.constant("Hello!")
print(hello)

hello_out = sees.run(hello)
print_tf(hello_out)

weight = tf.Variable(tf.random_normal_initializer)


"""