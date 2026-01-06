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

    
    def forwardProp(self,X:np.ndarray) -> tuple:
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

        m = y.shape[0] # m is the size of batch
        dz2 = a2 - y.T
        dw2 = 1/m * np.dot(dz2,a1.T) # dot because we calculate for the whole batch
        db2 = 1/m * np.sum(dz2, axis=1, keepdims=True) # sum because we calculate for the whole batch
        dz1 = np.dot(self.w2.T,dz2) * ReLU_m(z1)
        dw1 = 1/m * np.dot(dz1,X)
        db1 = 1/m * np.sum(dz1, axis=1, keepdims=True)

        return dw1,db1,dw2,db2


    
    def update_parameters(self, dw1, db1, dw2, db2, alpha) -> None:
        """
        Update parameters
        """
        self.w1 -= alpha * dw1
        self.b1 -= alpha * db1
        self.w2 -= alpha * dw2
        self.b2 -= alpha * db2
    
    

    def stocasticGradientDescent(self, X: np.ndarray, y: np.ndarray, epochs: int, alpha: float, beta: float = 0.9) -> np.ndarray:
        '''
        Gradient Descent
        '''
        accuracy_arr = []
        for epoch in range(epochs):
            z1, a1, z2, a2 = self.forwardProp(X)
            dw1, db1, dw2, db2 = self.backProp(z1, a1, z2, a2, X, y)
            self.momentum(dw1, db1, dw2, db2, alpha)
            
            if epoch % 10 == 0:
                predictions = self.predict(X)
                accuracy = get_accuracy(predictions, y)
                accuracy_arr.append(accuracy)
                print(f'Epoch: {epoch}, accuracy: {accuracy:.4f}')
        return np.array(accuracy_arr)

    def batchGradientDescent(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size : int, alpha: float, beta: float = 0.9) -> np.ndarray:
        '''
        batch gradient descent 
        '''
        accuracy_arr = []
        for epoch in range(epochs):
            for i in np.arange(start=0, stop=X.shape[0], step = batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                z1, a1, z2, a2 = self.forwardProp(X_batch)
                dw1, db1, dw2, db2 = self.backProp(z1, a1, z2, a2, X_batch, y_batch)
                self.rmsprop(dw1, db1, dw2, db2, alpha)
            
            if epoch % 5 == 0:
                predictions = self.predict(X)
                accuracy = get_accuracy(predictions, y)
                accuracy_arr.append(accuracy)
                print(f'Epoch: {epoch}, accuracy: {accuracy:.4f}')
        return np.array(accuracy_arr)
        
    def momentum(self, dw1, db1, dw2, db2, learning_rate, beta=0.9):
        """
        stadard momentum optimizer for gradient descent
        """

        if not hasattr(self, "m_w1"):
            self.m_w1 = np.zeros(self.w1.shape)
            self.m_b1 = np.zeros(self.b1.shape)
            self.m_w2 = np.zeros(self.w2.shape)
            self.m_b2 = np.zeros(self.b2.shape)

        self.m_w1 = beta * self.m_w1 + (1 - beta) * dw1 
        self.m_b1 = beta * self.m_b1 + (1 - beta) * db1 
        self.m_w2 = beta * self.m_w2 + (1 - beta) * dw2
        self.m_b2 = beta * self.m_b2 + (1 - beta) * db2 

        self.w1 -= self.m_w1 * learning_rate
        self.b1 -= self.m_b1 * learning_rate
        self.w2 -= self.m_w2 * learning_rate
        self.b2 -= self.m_b2 * learning_rate

    def rmsprop(self, dw1, db1, dw2, db2, learning_rate, beta=0.9, epsilon=1e-8):
        """
        RMSprop optimizer
        RMSprop uses adaptive learning rate
        """

        if not hasattr(self, "v_w1"):
            self.v_w1 = np.zeros(self.w1.shape)
            self.v_b1 = np.zeros(self.b1.shape)
            self.v_w2 = np.zeros(self.w2.shape)
            self.v_b2 = np.zeros(self.b2.shape)

        self.v_w1 = beta * self.v_w1 + (1 - beta) * dw1 ** 2
        self.v_b1 = beta * self.v_b1 + (1 - beta) * db1 ** 2
        self.v_w2 = beta * self.v_w2 + (1 - beta) * dw2 ** 2
        self.v_b2 = beta * self.v_b2 + (1 - beta) * db2 ** 2
        

        self.w1 -= learning_rate / np.sqrt(self.v_w1 + epsilon) * dw1 
        self.b1 -= learning_rate / np.sqrt(self.v_b1 + epsilon) * db1 
        self.w2 -= learning_rate / np.sqrt(self.v_w2 + epsilon) * dw2 
        self.b2 -= learning_rate / np.sqrt(self.v_b2 + epsilon) * db2 

            
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
    return np.where(x > 0, 1, 0)


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
