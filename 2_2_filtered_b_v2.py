# kaihao, 2023/1/31 in wuhan
# happy birthday to my older sis whose birthday wish is that everybody stays safe and healthy!
# cannot believe January has already ended
# On the basis of 4_0: set a batch of Thresholds and BinIntervals, check their sensitivity

# kaihao, 2023/3/14 in Yaogan Building with fever
# Finally starting to fix the algorithm
# Based on global csv files of reference datasets by 6_linear_fit/3_global_csv.py

# kaihao, 2023/11/28
# re-estimate b on the adjusted reference dataset, now we only need i=20 with various thresholds

# kaihao, 2023/11/29
# make a iterative-filtering algorithm

import sys
import glob
import pandas as pd
import numpy as np
from itertools import product # for Descrates Product
from math import floor, ceil
from numba import jit

# Threshold = 80 # percentile
# BinInterval = 10 # interval = 1/n
Thresholds = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
BinInterval = 100
hand_threshold = 20
choices = list(product([BinInterval], Thresholds))
b_choices = [f'b_i{i}_t{j}' for i,j in choices]

baseline = -4.605 # that's ln(0.01)

# find numbers
numbers = []
with open('1_available_basins.txt') as f:
    for line in f:
        numbers.append(line.strip())
# print(numbers)

# initializing results

empty_count = 0
df_results = pd.DataFrame(columns = b_choices)
df_results.insert(loc = 0, column = 'id', value = '')
# breakpoint()

for number in numbers:

    print(f'Processing basin {number}...')

    # read file
    df_a = pd.read_csv(f'/home/zhengkh/zkh/floodplain_lev3/no_{number}/csv_for_optimization_update_v1.csv')
    
    df_try_a = df_a.loc[(df_a['ref'] != 0) & (df_a['hnd'] != 0)].sort_values('hnd', ascending=True)
    df_try_a = df_try_a.loc[(df_try_a['hnd'] < hand_threshold)]
    df_try_a['ln_hnd'] = np.log(df_try_a['hnd'])
    df_try_a['ln_upa'] = np.log(df_try_a['upa_3']**(10/3))

    df_intersection = df_try_a[['hnd','ln_upa','ref']].loc[df_try_a['ref']==3]
    df_intersection = df_intersection.rename(columns={'ln_upa':'x','hnd':'y'}) 

    # check if file is available (some files have 0 lines)
    if len(df_intersection) == 0:
        # print(f'Processing basin {number}, Dataframe empty')
        empty_count += 1
        df_results = df_results.append(pd.DataFrame({'id':[number]}))
        continue


    x_min = floor(df_intersection['x'].min())
    x_max = ceil(df_intersection['x'].max())


    # moving-window filtering
    df_intersection = df_intersection[['y','x']]
    data_array = df_intersection.to_numpy()

    def update_sliding_window_stats(data, x_min, x_max):
        window_stats = []
        for start in np.arange(x_min, x_max, 0.1):
            end = start + 1
            window_df = data[(data[:, 1] >= start) & (data[:, 1] <= end)]
            mean = np.mean(window_df[:, 0])
            std = np.std(window_df[:, 0])
            window_stats.append((start, end, mean, std))
        return np.array(window_stats)

    @jit(nopython=True)
    def filter_data_with_stats(data, window_stats):
        keep = np.ones(len(data), dtype=np.bool_)
        for index in range(len(data)):
            x_val = data[index, 1]
            y_val = data[index, 0]
            for start, end, mean, std in window_stats:
                if start <= x_val <= end:
                    lower_bound = mean - 3 * std
                    upper_bound = mean + 3 * std
                    if not (lower_bound <= y_val <= upper_bound):
                        keep[index] = False
                        break
        return data[keep]

    def iterate_filtering(data):
        x_min, x_max = data[:, 1].min(), data[:, 1].max()
        data_removed = True
        while data_removed:
            window_stats = update_sliding_window_stats(data, x_min, x_max)
            new_data = filter_data_with_stats(data, window_stats)
            if len(new_data) < len(data):
                print(f"Filtered {len(data)-len(new_data)} data points, {len(new_data)} remaining.")
                data = new_data
                data_removed = True     
            else:
                data_removed = False
        return data

    # 使用外层函数进行迭代过滤
    filtered_data = iterate_filtering(data_array)

    # 转换回Pandas DataFrame
    df_intersection = pd.DataFrame(filtered_data, columns=['y', 'x'])    
    df_intersection['y'] = np.log(df_intersection['y'])
    
    result_b_list = []

    # sort x into bins
    df_intersection['b'] = (df_intersection['y']-baseline)/df_intersection['x'] # assume y = bx + _baseline
    df_intersection['x_bin'] = round(df_intersection['x']*BinInterval)/BinInterval
    df1 = df_intersection.groupby('x_bin')['b'].agg(max)
    list_b = df1.tolist()
    list_b = sorted(list_b)

    # get percentile
    array_b = np.array(list_b)
    # breakpoint()

    for Threshold in Thresholds:
        b = np.percentile(array_b,100-Threshold)
        result_b_list.append(b)
        # print(f'Processing basin {number}, b = {b}')

    df_temp = pd.DataFrame({'id':[number]})
    for index,(i,j) in enumerate(choices):
        df_temp[f'b_i{i}_t{j}']=result_b_list[index]
        # print(df_temp)
    df_results = df_results.append(df_temp, ignore_index = True)
    # breakpoint()

df_results.to_csv(f'optimize_b/b_optimize_total_v3.csv',index=False,header=True)
print(f'Finished processing, {len(numbers)} files in total with {empty_count} empty.')