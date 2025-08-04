import numpy as np

class NeuralNetwork:
    def __init__(self,hidden_size) -> None:
        '''
        Initalizere parameters (weights and biases)
        '''
        self.w1 = np.random.randn(hidden_size, 784) 
        self.b1 = np.random.randn(hidden_size, 1)  
        self.w2 = np.random.randn(10, hidden_size)
        self.b2 = np.random.randn(10, 1) 
    
    def forwardProp(self,X:np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        '''
        Forward propagation
        '''
        z1 = np.dot(self.w1,X.T) + self.b1
        a1 = ReLU(z1)
        z2 = np.dot(self.w2,a1) + self.b2
        a2 = softmax(z2)

        return z1,a1,z2,a2
    
    def backProp(self,z1,a1,z2,a2,X,y) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """
        Backpropagation
        using categoricalcrossentropy loss function which is simplyfied A2 - y

        the reason for sum and dotproduct is because X is a whole batch
        """

        m = y.shape[0]
        dz2 = a2 - y.T
        dw2 = 1/m * np.dot(dz2,a1.T) # dot because we calculate for the whole batch
        db2 = 1/m * np.sum(dz2, axis=1, keepdims=True) # sum because we calculate for the whole batch
        dz1 = np.dot(self.w2.T,dz2) * ReLU_m(z1)
        dw1 = 1/m * np.dot(dz1,X)
        db1 = 1/m * np.sum(dz1, axis=1, keepdims=True)

        return dw1,db1,dw2,db2


    
    def update_parameters(self,alpha,dw1,db1,dw2,db2) -> None:
        """
        Update parameters
        """
        self.w1 -= alpha * dw1
        self.b1 -= alpha * db1
        self.w2 -= alpha * dw2
        self.b2 -= alpha * db2
    


    def gradientDescent(self, X: np.ndarray, y: np.ndarray, epochs: int, alpha: float) -> None:
        '''
        Gradient Descent
        '''
        accuracy_arr = []
        for epoch in range(epochs):
            z1, a1, z2, a2 = self.forwardProp(X)
            dw1, db1, dw2, db2 = self.backProp(z1, a1, z2, a2, X, y)
            self.update_parameters(alpha, dw1, db1, dw2, db2)
            
            if epoch % 10 == 0:
                predictions = self.predict(X)
                accuracy = get_accuracy(predictions, y)
                accuracy_arr.append(accuracy)
                print(f'Epoch: {epoch}, accuracy: {accuracy:.4f}')
        return np.array(accuracy_arr)
        


            
    def predict(self,X:np.ndarray) -> np.ndarray:
        '''
        changes from a onehot encoded vector to a number
        outputs matrix (10,)
        '''
        _, _, _, a2 = self.forwardProp(X)
        return np.argmax(a2, axis=0)

def ReLU_m(x:float) -> float:
    # derivative of relu funktion
    # if x > 0 return 1 else 0
    return np.where(x > 0, 1,0)


def softmax(x:float) -> float:
    # Softmax functionen 
    # returns probability distribution
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def ReLU(x:float) -> float:
    # ReLu functionen
    return np.maximum(0,x)

def get_accuracy(predictions: np.ndarray, y: np.ndarray) -> float:
    true_labels = np.argmax(y, axis=1)
    return np.mean(predictions == true_labels)
