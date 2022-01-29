# 6.2.2 은닉층의 활성화값 분포
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    mask = x <= 0
    x[mask] = 0
    return x

x = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    
    # w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01 # 가중치의 표준편차 = 0.01
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num) # Xavier 초기값 : 가중치의 표준편차 = 1/sqrt(n)
    # w = np.random.randn(node_num, node_num) * (np.sqrt(2 / node_num)) # He 초기값 (ReLU에 특화된 초기값) : 가중치의 표준편차 = sqrt(2/n)
    a = np.dot(x, w)
    # z = sigmoid(a)
    z = relu(a)
    activations[i] = z

for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + '-layer')
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()