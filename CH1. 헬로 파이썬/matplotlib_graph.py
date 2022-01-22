import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# 1.6.1 단순한 그래프 그리기
x = np.arange(0, 6, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.show()

# 1.6.2 pyplot의 기능
x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label='sin')
plt.plot(x, y2, linestyle='--', label='cos')
plt.xlabel('x') # x축 이름
plt.ylabel('y') # y축 이름
plt.title('sin & cos') # 제목
plt.legend()
plt.show()

# 1.6.3 이미지 표시하기
img = imread('../dataset/cactus.png')
plt.imshow(img)
plt.show()