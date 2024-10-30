import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def main():
    FILEPATH = 'Data/mnist_data.csv'
    X,y = load_data(FILEPATH)
    X_train,y_train,X_test,y_test = split_data(X,y)
    plot_numbers(X_test)
    #return X_train,y_train,X_test,y_test

def load_data(path:str) -> tuple[list,list]:
    train_df = pd.read_csv(path)

    y = train_df.loc[:,'label']
    X = train_df.drop('label',axis =1)

    return X.to_numpy(),y.to_numpy()

def split_data(X:pd.DataFrame,y:pd.DataFrame) -> tuple[list,list,list,list]:
    X_train,y_train,X_test,y_test = train_test_split(X,y,test_size=0.2,random_state=4,stratify=y)
    return X_train,y_train,X_test,y_test

def plot_numbers(num:np.ndarray) -> None:
    # Plotting numbers
    print(num.shape)
    #image = num.reshape((28,28))
    #fig = plt.figure
    #plt.imshow(num, cmap='gray')
    #plt.show()

if __name__ == '__main__':
    main()