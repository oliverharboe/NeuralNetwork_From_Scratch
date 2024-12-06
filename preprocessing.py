import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from neuralnetwork import NeuralNetwork

def main():
    FILEPATH = 'Data/mnist_data.csv'
    X,y = load_data(FILEPATH)
    X_train,y_train,X_test,y_test = split_data(X,y)
    y_train = oneHotlabel(y_train)
    model = NeuralNetwork()
    model.gradientDescent(X_train,y_train,epochs=100,alpha=0.0001)

def load_data(path:str) -> tuple[np.ndarray,np.ndarray]:
    '''
    loading data from csv
    '''
    train_df = pd.read_csv(path)

    y = train_df.loc[:,'label']
    X = train_df.drop('label',axis=1)

    return X.to_numpy(),y.to_numpy()

def split_data(X:pd.DataFrame,y:pd.DataFrame) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    '''
    splits data into train and test
    '''
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4,stratify=y)
    return X_train,y_train,X_test,y_test

def normalize_data(data):
    # change interval from [0;255] to [0;1]
    return data/255

def oneHotlabel(y:np.ndarray) -> np.ndarray:
    """
    Onehot encode labels output shape is n * 10
    where n is the number of labels
    arange runs through the rows. then it indexes to the y[i] value og places a 1 in that place

    """
    oneHot = np.zeros((y.shape[0],10))
    oneHot[np.arange(y.shape[0]),y] = 1
    return oneHot

def plot_numbers(num:np.ndarray,label:np.ndarray = None) -> None:
    '''
    Creating a gray scale image of the number (with and without label)
    '''
    print(num.shape)
    image = num.reshape(28,28)
    plt.imshow(image, cmap='gray')
    if label != None:
        plt.title(f'Label: {label}')
    plt.show()

if __name__ == '__main__':
    main()