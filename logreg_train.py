import argparse
import os
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='path to data.csv', default='data/dataset_train.csv')
    parser.add_argument('-o', '--output', type=str, help='path to thetas.csv', default='data/thetas.csv')
    # parser.add_argument('-s', '--show', type=int, help='show plot', default=False)
    # parser.add_argument('-r', '--r_squared', type=int, help='show R2 metric', default=False)
    parser.add_argument('-l', '--learning_rate', type=float, help='set learning rate', default=0.1)
    parser.add_argument('-e', '--epochs', type=float, help='set epoch number', default=1000)
    args = parser.parse_args()
    return args.__dict__


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_cost(df, theta, y):
    df_len = len(df)
    sig = sigmoid(np.dot(theta, df.T))
    cost = (np.sum((y.T * np.log(sig)) + ((1 - y.T) * (np.log(1 - sig))))) / -df_len
    derivative = (np.dot((sig - y.T), df)) / df_len
    return derivative, cost


def gradient_descent(df, y, epochs, learning_rate):
    history = []
    theta = np.zeros((1, df.shape[1]))
    for _ in range(epochs):
        derivative, cost = get_cost(df, theta, y)
        theta -= learning_rate * derivative
        history.append(cost)
    return theta[0].tolist(), history


def std_scaler(df):
    return (df - df.mean()) / df.std()


def train(df, epochs, learning_rate):
    if df is None:
        return None
    houses = {
        'Ravenclaw': 0,
        'Slytherin': 1,
        'Gryffindor': 2,
        'Hufflepuff': 3
    }
    history_dct = {}
    thetas = []
    houses_indexes = [houses[x] for x in df['Hogwarts House']]
    df = df.iloc[:, 6:]
    df = df.replace(np.nan, 0)
    df = std_scaler(df)
    for i, val in enumerate(houses):
        y = []
        for house in houses_indexes:
            y.append(1 if house == i else 0)
        theta, history = gradient_descent(df, np.asarray(y), epochs, learning_rate)
        history_dct[val] = history
        thetas.append(theta)
    thetas = pd.DataFrame(thetas, columns=df.columns, index=houses)
    return thetas


def main():
    args = parse_args()
    if os.path.exists(args['input']):
        try:
            df = pd.read_csv(args['input'])
            thetas = train(df, args['epochs'], args['learning_rate'])
            if thetas is not None:
                thetas.to_csv(args['output'])
        except OSError as e:
            print('Cannot open file:', e)
        except Exception as e:
            print('Unknown error:', e)
    else:
        exit('False path to data.csv. Try: \'python3 logreg_train.py --path data/dataset_train.csv\'')


if __name__ == '__main__':
    main()
