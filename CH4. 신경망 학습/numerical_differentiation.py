# 4.3.1 미분
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

# 4.3.2 수치 미분의 예
def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

import numpy as np
import matplotlib.pylab as plt

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, y)
plt.show()

print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))

# 4.3.3 편미분
def function_2(x):
    return x[0] ** 2 + x[1] ** 2

def function_tmp1(x0):
    return x0 * x0 + 4.0 ** 2

def function_tmp2(x1):
    return 3.0 * 3.0 + x1 ** 2

print(numerical_diff(function_tmp1, 3.0))
print(numerical_diff(function_tmp2, 4.0))