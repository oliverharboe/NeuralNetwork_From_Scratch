import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    FILEPATH = 'Data/mnist_data.csv'
    X,y = load_data(FILEPATH)
    X_train,y_train,X_test,y_test = split_data(X,y)
    return X_train,y_train,X_test,y_test

def load_data(path):
    train_df = pd.read_csv(path)

    y = train_df.loc[:,'label']
    X = train_df.drop('label',axis =1)

    return X.to_numpy(),y.to_numpy()

def split_data(X,y):
    X_train,y_train,X_test,y_test = train_test_split(X,y,test_size=0.2,random_state=4,stratify=y)
    return X_train,y_train,X_test,y_test


if '__name__' == main:
    main()
