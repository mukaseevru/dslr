import argparse
import math
import os
import pandas as pd
pd.options.display.float_format = '{:.5f}'.format


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='path to data.csv', default='data/dataset_train.csv')
    args = parser.parse_args()
    return args.__dict__


def describe_count(lst):
    lst = [x for x in lst if not pd.isna(x)]
    return len(lst)


def describe_mean(lst):
    lst = [x for x in lst if not pd.isna(x)]
    return sum(lst) / len(lst)


def describe_std(lst):
    lst = [x for x in lst if not pd.isna(x)]
    sum_lst = 0.0
    count_lst = describe_count(lst) - 1
    mean_lst = describe_mean(lst)
    for elem in lst:
        sum_lst += (elem - mean_lst) ** 2
    return (sum_lst / count_lst) ** 0.5


def describe_min(lst):
    lst = [x for x in lst if not pd.isna(x)]
    min_lst = lst[0]
    for elem in lst:
        if elem < min_lst:
            min_lst = elem
    return min_lst


def describe_max(lst):
    lst = [x for x in lst if not pd.isna(x)]
    max_lst = lst[0]
    for elem in lst:
        if elem > max_lst:
            max_lst = elem
    return max_lst


def describe_percent(lst, percent):
    lst = [x for x in lst if not pd.isna(x)]
    lst = sorted(lst)
    k = (len(lst) - 1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return lst[int(k)]
    d0 = lst[int(f)] * (c - k)
    d1 = lst[int(c)] * (k - f)
    return d0 + d1


def describe(df):
    df = df.select_dtypes(include=['int64', 'float64'])
    table = pd.DataFrame(columns=df.columns,
                         index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    for column in df.columns:
        table.loc['count', column] = describe_count(df[column].values)
        table.loc['mean', column] = describe_mean(df[column].values)
        table.loc['std', column] = describe_std(df[column].values)
        table.loc['min', column] = describe_min(df[column].values)
        table.loc['25%', column] = describe_percent(df[column].values, 0.25)
        table.loc['50%', column] = describe_percent(df[column].values, 0.5)
        table.loc['75%', column] = describe_percent(df[column].values, 0.75)
        table.loc['max', column] = describe_max(df[column].values)
        table[column] = table[column].astype('float64')
    return table


def main():
    args = parse_args()
    if os.path.exists(args['path']):
        try:
            df = pd.read_csv(args['path'])
            print(describe(df))
        except OSError as e:
            print('Cannot open file:', e)
        except Exception as e:
            print('Unknown error:', e)
    else:
        exit('False path to data.csv. Try: \'python3 describe.py --path data/dataset_train.csv\'')


if __name__ == '__main__':
    main()
