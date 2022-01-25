# 5.4.1 곱셈 계층
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    
    def backward(self, dout):
        dx = dout * self.y # 상류에서 넘어온 미분(dout)에 순전파 때의 값을 서로 바꿔 곱한 후 하류로 흘린다
        dy = dout * self.x
        return dx, dy

apple = 100
apple_num = 2
tax = 1.1

# 계층들
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
print(price)

# 역전파
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_num, dtax)

# 5.4.2 덧셈 계층
class AddLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

apple_num = 2
apple = 100
orange = 150
orange_num = 3
tax = 1.1

# 계층들
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_price_layer = AddLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple_num, apple)
orange_price = mul_orange_layer.forward(orange, orange_num)
price = add_price_layer.forward(apple_price, orange_price)
price_tax = mul_tax_layer.forward(price, tax)
print(price_tax)

# 역전파
dprice_tax = 1
dprice, dtax = mul_tax_layer.backward(dprice_tax)
dapple_price, dorange_price = add_price_layer.backward(dprice)
dapple_num, dapple = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
print(dapple_num, dapple, dorange, dorange_num, dtax)