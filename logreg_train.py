import argparse
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='path to data.csv', default='data/dataset_train.csv')
    parser.add_argument('-o', '--output', type=str, help='path to thetas.csv', default='data/thetas.csv')
    parser.add_argument('-s', '--show_plot', type=int, help='save plot to file', default=0)
    parser.add_argument('-l', '--learning_rate', type=float, help='set learning rate', default=0.1)
    parser.add_argument('-e', '--epochs', type=float, help='set epoch number', default=10000)
    parser.add_argument('-p', '--precision', type=float, help='set precision', default=0.000001)
    parser.add_argument('-m', '--method', type=str, help='set method GD, SGD, MBGD', default='GD')
    parser.add_argument('-b', '--batch_size', type=int, help='set batch size', default=2)
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


def get_mini_batches(df, y, batch_size):
    mini_batches = []
    n_mini_batches = df.shape[0] // batch_size
    for i in range(n_mini_batches):
        df_mini = df[i * batch_size: (i + 1) * batch_size]
        y_mini = y[i * batch_size: (i + 1) * batch_size]
        mini_batches.append((df_mini, y_mini))
    if df.shape[0] % batch_size != 0:
        df_mini = df[n_mini_batches * batch_size: -1]
        y_mini = y[n_mini_batches * batch_size: -1]
        mini_batches.append((df_mini, y_mini))
    return mini_batches


def gradient_descent(df, y, epochs, learning_rate, precision, method, mini_batches):
    history = []
    theta = np.zeros((1, df.shape[1]))
    cost_prev = 0
    cost = 1
    i = 0
    random_ind = 0
    while abs(cost - cost_prev) > precision and i < epochs:
        cost_prev = cost
        if method == 'SGD':
            derivative, cost = get_cost(df.iloc[[random_ind, ]], theta, y[random_ind])
            if random_ind == df.shape[0] - 1:
                random_ind = 0
            else:
                random_ind += 1
        elif method == 'MBGD':
            derivative, cost = get_cost(mini_batches[random_ind][0], theta, np.array(mini_batches[random_ind][1]))
            if random_ind == len(mini_batches) - 1:
                random_ind = 0
            else:
                random_ind += 1
        else:
            derivative, cost = get_cost(df, theta, y)
        theta -= learning_rate * derivative
        history.append(cost)
        i += 1
    if abs(cost - cost_prev) > precision:
        print('Error: Calculation stopped, maximum number of epochs exceeded.')
    else:           
        print('epochs = ', i)        
    return theta[0].tolist(), history


def std_scaler(df):
    return (df - df.mean()) / df.std()


def train(df, epochs=10000, learning_rate=0.1, precision=0.000001, method='GD', batch_size=2):
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
    if method in ('SGD', 'MBGD'):
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    houses_indexes = [houses[x] for x in df['Hogwarts House']]
    df.drop(df.columns[:6], axis=1, inplace=True)
    df = df.replace(np.nan, 0)
    df = std_scaler(df)
    mini_batches = None
    for i, val in enumerate(houses):
        print('Training the classifier for class k = {}...'.format(val))
        y = []
        for house in houses_indexes:
            y.append(1 if house == i else 0)
        if method == 'MBGD':
            mini_batches = get_mini_batches(df, y, batch_size)
        theta, history = gradient_descent(df, np.asarray(y), epochs, learning_rate, precision, method, mini_batches)
        history_dct[val] = history
        thetas.append(theta)
    thetas = pd.DataFrame(thetas, columns=df.columns, index=houses)
    print('Training is completed!')
    return thetas, history_dct


def plot(history_dct):
    sns.set_style('white')
    for house in history_dct:
        sns.scatterplot(x=range(len(history_dct[house])), y=history_dct[house], label=house)
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('training')
    plt.savefig('plots/training.png')
    print('Plot save to plots/training.png')


def main():
    args = parse_args()
    if os.path.exists(args['input']):
        try:
            df = pd.read_csv(args['input'])
            thetas, history_dct = train(df, args['epochs'], args['learning_rate'], args['precision'],
                                        args['method'], args['batch_size'])
            if thetas is not None:
                thetas.to_csv(args['output'])
            if history_dct is not None and args['show_plot'] == 1:
                plot(history_dct)
        except OSError as e:
            print('Cannot open file:', e)
        except Exception as e:
            print('Unknown error:', e)
    else:
        exit('False path to data.csv. Try: \'python3 logreg_train.py --path data/dataset_train.csv\'')


if __name__ == '__main__':
    main()
