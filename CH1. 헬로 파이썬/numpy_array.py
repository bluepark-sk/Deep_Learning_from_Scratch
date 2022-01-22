# 1.5.1 넘파이 가져오기
import numpy as np

# 1.5.2. 넘파이 배열 생성하기
x = np.array([1.0, 2.0, 3.0])
print(x)
type(x)

# 1.5.3 넘파이의 산술 연산
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
x + y
x - y
x * y
x / y

# 브로드캐스트 : 넘파이 배열과 스칼라값의 조합으로 된 산술 연산 수행. 스칼라값과의 계산이 넘파이 배열의 원소별로 한 번씩 수행.
x = np.array([1.0, 2.0, 3.0])
x / 2.0

# 1.5.4 넘파이의 N차원 배열
A = np.array([[1, 2], [3, 4]])
print(A)
A.shape
A.dtype

B = np.array([[3, 0], [0, 6]])
A + B
A * B

# 브로드캐스트 작동
print(A)
A * 10

# 1.5.5 브로드캐스트
A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
A * B

# 1.5.6 원소 접근
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
X[0]
X[0][1]

for row in X:
    print(row)

X = X.flatten()
print(X)
X[np.array([0, 2, 4])]

X > 15
X[X>15]