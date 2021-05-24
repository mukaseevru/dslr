import argparse
import pandas as pd
import numpy as np
import math
import os
pd.options.display.float_format = '{:.5f}'.format


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='path to data.csv', default='data/dataset_train.csv')
    args = parser.parse_args()
    return args.__dict__


def describe_count(lst):
    count = 0
    for elem in lst:
        if pd.isna(elem):
            continue
        count += 1
    return count


def describe_mean(lst):
    len_lst = len(lst)
    sum_lst = 0.0
    for elem in lst:
        if pd.isna(elem):
            len_lst -= 1
            continue
        sum_lst += elem
    return sum_lst / len_lst


def describe_std(lst):
    sum_lst = 0.0
    count_lst = describe_count(lst) - 1
    mean_lst = describe_mean(lst)
    for elem in lst:
        if pd.isna(elem):
            continue
        sum_lst += (elem - mean_lst) ** 2
    return (sum_lst / count_lst) ** 0.5


def describe_min(lst):
    min_lst = lst[0]
    for elem in lst:
        if pd.isna(elem):
            continue
        if elem < min_lst or pd.isna(min_lst):
            min_lst = elem
    return min_lst


def describe_max(lst):
    max_lst = lst[0]
    for elem in lst:
        if pd.isna(elem):
            continue
        if elem > max_lst or pd.isna(max_lst):
            max_lst = elem
    return max_lst


def describe_percent(lst, percent):
    lst = sorted(lst)
    k = (len(lst) - 1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return lst[int(k)]
    d0 = lst[int(f)] * (c-k)
    d1 = lst[int(c)] * (k-f)
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
        except OSError as e:
            print('Cannot open file:', e)
        except Exception as e:
            print('Unknown error:', e)
    else:
        exit('False path to data.csv. Try: \'python3 describe.py --path data/dataset_train.csv\'')
    print(describe(df))


if __name__ == '__main__':
    main()
