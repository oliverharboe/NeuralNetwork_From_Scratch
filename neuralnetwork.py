import numpy as np

class NerualNetwork:
    def __init__(self) -> None:
        '''
        Initalizere parameters (weights and biases)
        '''
        self.w1 = np.random.randn(16,784)
        self.b1 = np.random.randn(16,1)
        self.w2 = np.random.randn(10,16)
        self.b2 = np.random.randn(10,1)
    
    def forwardProp(self,X:np.ndarray) -> np.ndarray:
        '''
        Forward propagation
        '''
        z1 = np.dot(self.w1,X.T) + self.b1
        a1 = ReLU(z1)
        z2 = np.dot(self.w2,a1) + self.b2
        a2 = softmax(z2)

        return z1,a1,z2,a2
    
    def backProp(self,X:np.ndarray,y:np.ndarray) -> None:
        """
        Backpropagation
        """

    


    def gradientDescent(self,X:np.ndarray,y:np.ndarray,epochs:int,learning_rate:float) -> None:
        '''
        Gradient Descent
        optimizing the parameters
        '''
        for epoch in range(epochs):
            pass
            
    def prediction(self,X:np.ndarray) -> np.ndarray:
        '''
        changes from a onehot encoded vector to a number
        '''
        vector = self.forwardProp(X)
        pre = np.argmax(vector,axis=0)
        return pre

def ReLU_m(x:float) -> float:
    # differentieret relu funktion
    return np.where(x > 0, 1,0)


def softmax(x:float) -> float:
    # Softmax funktionen 
    # funktionen er en variant der gør at 
    
    y = np.exp(x-np.max(x)) / np.sum(np.exp(x-np.max(x)),axis=0)
    return y

def ReLU(x:float) -> float:
    # Definere ReLu funktionen
    return np.maximum(0,x)
