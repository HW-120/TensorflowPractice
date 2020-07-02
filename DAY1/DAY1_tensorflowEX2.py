import tensorflow as tf
import numpy as np


a = [[1.0, 2.0], [3.0, 4.0]]
b = np.array([[1.0, 2.0],
              [3.0, 4.0]])
c = tf.constant([[1.0, 2.0],[3.0, 4.0]])

print(type(a))
print(tf.convert_to_tensor(a, dtype=tf.float32))

print()
ndarray = np.ones([3, 3])  # 3행 3열에 1을 집어 넣음
tensor = tf.multiply(ndarray, 42) # 안에 모두 42로 변환 / 연동이 좋음
print(ndarray)
print(tensor)
print(c.numpy())

print()
weight = tf.Variable(tf.random_normal_initializer(stddev=0.1)([5,2]))
print(weight)

print()
npp = np.arange(10)
print(npp)
ds_tensors = tf.data.Dataset.from_tensor_slices(npp) # 데이터셋을 만들어줌

 # map은 함수를 정의 / batch 사이즈가 2 = 데이터는 두 개씩 공급
ds_tensors = ds_tensors.map(tf.square).shuffle(20).batch(2)

#데이터 공급
print('Elements of ds_tensors:')
for _ in range(3):  # 셔플이 에폭마다 잘 되는지 확인하기 위해 (?)
    for x in ds_tensors.take(2): # 데이터셋을 바로 활용 가능
        print(x)
# 한 번 끝난 다음에 셔플이 돌아서 한 번 나온 데이터는 모든 데이터가 다 나올 때까지 안나옴
