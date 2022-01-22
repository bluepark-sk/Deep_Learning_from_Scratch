print("I'm hungry!")

# 1.4.2 클래스
class Man:
    def __init__(self, name) -> None:
        self.name = name
        print("Initialized!")
    
    def hello(self):
        print("Hello " + self.name + "!")
    
    def goodbye(self):
        print("Good-bye " + self.name + "!")

m = Man("David")
m.hello()
m.goodbye()