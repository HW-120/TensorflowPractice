import tensorflow as tf
# shape 같은 사소한 것에서 실수하지 않게 잘 확인
print(tf.random.normal(shape=(2,2), mean = 0., stddev=1.))

print()
print(tf.random.uniform(shape=(2,2), minval=0, maxval=10, dtype='int32'))

print()
initial_value = tf.random.normal(shape=(2,2))
a = tf.Variable(initial_value)
print(a)

print()     #gradient는 미분
a = tf.random.normal(shape=(2, 2))
b = tf.random.normal(shape=(2, 2))
with tf.GradientTape() as tape:
    tape.watch(a)   # 뭔가를 a로 미분할 거야 / 아래 애들의 관계를 계속 기록
    c = tf.sqrt(tf.square(a) + tf.square(b))
    dc_da = tape.gradient(c, a)
    print(dc_da)

