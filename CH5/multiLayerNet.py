# 5.7.2 오차역전파법을 적용한 신경망 구현하기
import sys, os
sys.path.append(os.pardir)
import numpy as np
from .affineLayer_softmaxLayer import Affine, SoftmaxWithLoss
from .reluLayer_sigmoidLayer import Relu
from common.gradient import numerical_gradient
from collections import OrderedDict

class MultiLayerNet:
    def __init__(self, input_size, hidden_size_list, output_size, weight_init_std=0.01):
        self.hidden_size_list = hidden_size_list

        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size_list[0])
        self.params['b1'] = np.zeros(hidden_size_list[0])
        for i in range(1, len(hidden_size_list)):
            self.params['W'+str(i+1)] = weight_init_std * np.random.randn(hidden_size_list[i-1], hidden_size_list[i])
            self.params['b'+str(i+1)] = np.zeros(hidden_size_list[i])
        self.params['W'+str(len(hidden_size_list)+1)] = weight_init_std * np.random.randn(hidden_size_list[-1], output_size)
        self.params['b'+str(len(hidden_size_list)+1)] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        for i in range(len(hidden_size_list)+1):
            self.layers['Affine'+str(i+1)] = Affine(self.params['W'+str(i+1)], self.params['b'+str(i+1)])
            self.layers['Relu'+str(i+1)] = Relu()
        self.lastLayer = SoftmaxWithLoss()
    
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def gradient(self, x, t):
        # 순전파
        self.loss(x, t)

        # 역전파
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        # 결과 저장
        grads = {}
        for i in range(len(self.hidden_size_list)+1):
            grads['W'+str(i+1)] = self.layers['Affine'+str(i+1)].dW
            grads['b'+str(i+1)] = self.layers['Affine'+str(i+1)].db
        return grads