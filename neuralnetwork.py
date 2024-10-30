import numpy as np

class NerualNetwork:
    def __init__(self) -> None:
        '''
        Initalizere parameters (weights and biases)
        '''
        self.w1 = np.random.randn(16,784)
        self.b1 = np.random.randn(16,1)
        self.w2 = np.random.randn(16,10)
        self.b2 = np.random.randn(10,1)
    
    def predict(self,X:np.ndarray) -> np.ndarray:
        '''
        Forward propagation
        '''
        z1 = np.dot(self.w1,X.T) + self.b1
        a1 = relu(z1)
        z2 = np.dot(self.w2,a1) + self.b2
        a2 = softmax(z2)
        return a2
    
    def fit(self,X:np.ndarray,y:np.ndarray) -> None:
        """
        Backpropagation
        """
        pass

def relu_m(x:float) -> float:
    # differentieret relu funktion
    return np.where(x > 0, 1,0)

    

def softmax(x:float) -> float:
    # Definere Softmax funktionen
    return np.exp(x) / np.sum(np.exp(x),axis=0)


def relu(x:float) -> float:
    # Definere ReLu funktionen
    return np.maximum(0,x)
