# 3.3.1 다차원 배열
import numpy as np
A = np.array([1, 2, 3, 4])
print(A)
np.ndim(A)
A.shape
A.shape[0]

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
np.ndim(B)
B.shape

# 3.3.2 행렬의 곱
A = np.array([[1, 2], [3, 4]])
A.shape
B = np.array([[5, 6], [7, 8]])
B.shape
np.dot(A, B)

A = np.array([[1, 2, 3], [4, 5, 6]])
A.shape
B = np.array([[1, 2], [3, 4], [5, 6]])
B.shape
np.dot(A, B)

A = np.array([[1, 2], [3, 4], [5, 6]])
A.shape
B = np.array([7, 8])
B.shape
np.dot(A, B)

# 3.3.3 신경망에서의 행렬 곱
X = np.array([1, 2])
X.shape
W = np.array([[1, 3, 5], [2, 4, 6]])
print(W)
W.shape
Y = np.dot(X, W)
print(Y)