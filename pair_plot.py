import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='path to data.csv', default='data/dataset_train.csv')
    args = parser.parse_args()
    return args.__dict__


def pair_plot(df):
    df = df.select_dtypes(include=['int64', 'float64']).drop('Index', axis=1).dropna()
    sns.pairplot(df, diag_kws={'bins': 10})
    # Добавить цвет, если будет время
    plt.savefig('plots/pair_plots.png')
    print('You can see pair plots here: plots/pair_plots.png')


def main():
    args = parse_args()
    if os.path.exists(args['path']):
        try:
            df = pd.read_csv(args['path'])
            pair_plot(df)
        except OSError as e:
            print('Cannot open file:', e)
        except Exception as e:
            print('Unknown error:', e)
    else:
        exit('False path to data.csv. Try: \'python3 describe.py --path data/dataset_train.csv\'')


if __name__ == '__main__':
    main()
