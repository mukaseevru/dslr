import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='path to data.csv', default='data/dataset_train.csv')
    args = parser.parse_args()
    return args.__dict__


def histogram(df):
    features = df.select_dtypes(include=['int64', 'float64']).drop('Index', axis=1).columns
    houses = ['Slytherin', 'Gryffindor', 'Ravenclaw', 'Hufflepuff']
    houses_dct = {}
    for house in houses:
        temp = df[df['Hogwarts House'] == house]
        houses_dct[house] = temp.select_dtypes(include=['int64', 'float64']).dropna()
    fig, axes = plt.subplots(4, 4, figsize=(15, 10))
    fig.suptitle('Histograms')
    i = 0
    for feature in features:
        i += 1
        plt.subplot(5, 3, i)
        plt.hist(houses_dct['Slytherin'][feature], 50, density=True,
                 color='red', alpha=0.75, label='Slytherin')
        plt.hist(houses_dct['Gryffindor'][feature], 50, density=True,
                 color='blue', alpha=0.75, label='Gryffindor')
        plt.hist(houses_dct['Ravenclaw'][feature], 50, density=True,
                 color='green', alpha=0.75, label='Ravenclaw')
        plt.hist(houses_dct['Hufflepuff'][feature], 50, density=True,
                 color='orange', alpha=0.75, label='Hufflepuff')
        plt.title(feature)
        plt.legend(loc='best')
    fig.tight_layout()
    plt.savefig('plots/histograms.png')
    print('You can see histograms here: plots/histograms.png')


def main():
    args = parse_args()
    if os.path.exists(args['path']):
        try:
            df = pd.read_csv(args['path'])
            histogram(df)
        except OSError as e:
            print('Cannot open file:', e)
        except Exception as e:
            print('Unknown error:', e)
    else:
        exit('False path to data.csv. Try: \'python3 describe.py --path data/dataset_train.csv\'')


if __name__ == '__main__':
    main()
