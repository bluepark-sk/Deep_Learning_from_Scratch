# 3.2.2 계단 함수 구현하기
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

import numpy as np
x = np.array([-1.0, 1.0, 2.0])
print(x)
y = x > 0
print(y)
y = y.astype(np.int)
print(y)

def step_function(x):
    y = x > 0
    return y.astype(np.int)

# 3.2.3 계단 함수의 그래프
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# 3.2.4 시그모이드 함수 구현하기
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# 3.2.7 ReLU 함수
def relu(x):
    return np.maximum(0, x)