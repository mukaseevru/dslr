import argparse
import os
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, help='path to data.csv', default='data/dataset_test.csv')
    parser.add_argument('-t', '--thetas', type=str, help='path to thetas.csv', default='data/thetas.csv')
    parser.add_argument('-o', '--output', type=str, help='path to houses.csv', default='data/houses.csv')
    # parser.add_argument('-s', '--show', type=int, help='show steps', default=False)
    args = parser.parse_args()
    return args.__dict__


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def std_scaler(df):
    return (df - df.mean()) / df.std()


def predict(df, thetas):
    if df is None or thetas is None:
        return None
    houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    thetas.drop(thetas.columns[0], axis=1, inplace=True)
    df = df.iloc[:, 6:]
    df = df.replace(np.nan, 0)
    df = std_scaler(df)
    result = {}
    for i, row in thetas.iterrows():
        sig = sigmoid(df.dot(row))
        result[houses[i]] = sig
    result = pd.DataFrame(result)
    y_pred = pd.DataFrame([houses[r.argmax()] for _, r in result.iterrows()], columns=['Hogwarts House'])
    return y_pred


def main():
    args = parse_args()
    if os.path.exists(args['data']):
        if os.path.exists(args['thetas']):
            try:
                df = pd.read_csv(args['data'])
                thetas = pd.read_csv(args['thetas'])
                y_pred = predict(df, thetas)
                if y_pred is not None:
                    y_pred.index.name = 'Index'
                    y_pred.to_csv(args['output'])
            except OSError as e:
                print('Cannot open file:', e)
            except Exception as e:
                print('Unknown error:', e)
        else:
            exit('False path to thetas.csv. Try: \'python3 logreg_predict.py --path data/thetas.csv\'')
    else:
        exit('False path to data.csv. Try: \'python3 logreg_predict.py --data data/dataset_test.csv\'')


if __name__ == '__main__':
    main()
