from preprocessing import main
import numpy as np

class NerualNetwork:
    def __init__(self) -> None:
        w1 = np.random.randn(16,784)
        b1 = np.random.randn(16,1)
        w2 = np.random.randn(16,10)
        w2 = np.random.randn(10,1)
    

    


def softmax(x):


def relu(x):
    if x < 0:
        return 0
    else:
        return x