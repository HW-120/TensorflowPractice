import tensorflow as tf 
import random
import numpy as np
import matplotlib.pyplot as plt


input_dim = 2
output_dim = 1
learning_rate = 0.01

# 가중치 행렬
w = tf.Variable(tf.random.uniform(shape=(input_dim, output_dim)))
# 편향 벡터
b = tf.Variable(tf.zeros(shape=(output_dim,)))  # 초기화

def compute_predictions(features):
    return tf.matmul(features, w) + b   # 실제 모델 부분

def compute_loss(labels, predictions):
    return tf.reduce_mean(tf.square(labels - predictions))  # mean square

def train_on_batch(x, y):
    with tf.GradientTape() as tape:
        predictions = compute_predictions(x)  # output
        loss = compute_loss(y, predictions)
        dloss_dw, dloss_db = tape.gradient(loss, [w,b])
    w.assign_sub(learning_rate * dloss_dw)
    b.assign_sub(learning_rate * dloss_db)
    return loss   # 굳이 리턴 안해도 되고 확인을 위해 함

num_samples = 10000
negative_samples = np.random.multivariate_normal(
    mean=[0,3], cov=[[1, 0.5], [0.5, 1]], size=num_samples)
positive_samples = np.random.multivariate_normal(
    mean=[3,0], cov=[[1, 0.5], [0.5, 1]], size=num_samples)
features = np.vstack((negative_samples, positive_samples)).astype(np.float32)
labels = np.vstack((np.zeros((num_samples, 1), dtype='float32'),
                    np.ones((num_samples, 1), dtype='float32')))
plt.scatter(features[:, 0], features[:, 1], c=labels[:, 0])  