# 2.3.1 간단한 구현부터
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

AND(0, 0)
AND(1, 0)
AND(0, 1)
AND(1, 1)

# 2.3.2 가중치와 편향 도입
import numpy as np
x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7
w * x
np.sum(w*x)
np.sum(w*x) + b

# 2.3.3 가중치와 편향 구현하기
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

# 2.5.2. XOR 게이트 구현하기
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

XOR(0, 0)
XOR(1, 0)
XOR(0, 1)
XOR(1, 1)