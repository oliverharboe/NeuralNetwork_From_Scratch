import numpy as np

class NerualNetwork:
    def __init__(self) -> None:
        self.w1 = np.random.randn(16,784)
        self.b1 = np.random.randn(16,1)
        self.w2 = np.random.randn(16,10)
        self.b2 = np.random.randn(10,1)
    
    def forward_prop(self,X):
        z1 = np.dot(self.w1,X.T) + self.b1
        a1 = relu(z1)
        z2 = np.dot(self.w2,a1) + self.b2
        a2 = softmax(z2)
        return a2
    
    def back_prop(self):
        pass

    


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x),axis=0)


def relu(x):
    return np.maximum(0,x)
