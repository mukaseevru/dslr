import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='path to data.csv', default='data/dataset_train.csv')
    args = parser.parse_args()
    return args.__dict__


def scatter_plot(df):
    features = df.select_dtypes(include=['int64', 'float64']).drop('Index', axis=1).columns
    houses = ['Slytherin', 'Gryffindor', 'Ravenclaw', 'Hufflepuff']
    houses_dct = {}
    for house in houses:
        temp = df[df['Hogwarts House'] == house]
        houses_dct[house] = temp.select_dtypes(include=['int64', 'float64']).dropna()
    fig, axes = plt.subplots(13, 13, figsize=(60, 40))
    fig.suptitle('Scatter plots')
    i = 0
    for feature in features:
        for feature2 in features:
            i += 1
            plt.subplot(13, 13, i)
            plt.scatter(x=houses_dct['Slytherin'][feature],
                        y=houses_dct['Slytherin'][feature2], color='red', label='Slytherin')
            plt.scatter(x=houses_dct['Gryffindor'][feature],
                        y=houses_dct['Gryffindor'][feature2], color='blue', label='Gryffindor')
            plt.scatter(x=houses_dct['Ravenclaw'][feature],
                        y=houses_dct['Ravenclaw'][feature2], color='green', label='Ravenclaw')
            plt.scatter(x=houses_dct['Hufflepuff'][feature],
                        y=houses_dct['Hufflepuff'][feature2], color='orange', label='Hufflepuff')
            plt.xlabel(feature)
            plt.ylabel(feature2)
    fig.tight_layout()
    plt.savefig('plots/scatter_plots.png')
    print('You can see scatter plots here: plots/scatter_plots.png')


def main():
    args = parse_args()
    if os.path.exists(args['path']):
        try:
            df = pd.read_csv(args['path'])
            scatter_plot(df)
        except OSError as e:
            print('Cannot open file:', e)
        except Exception as e:
            print('Unknown error:', e)
    else:
        exit('False path to data.csv. Try: \'python3 describe.py --path data/dataset_train.csv\'')


if __name__ == '__main__':
    main()
