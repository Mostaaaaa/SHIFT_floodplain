################################################################
################################################################
##                                                             #
##                 **Core Codes for SHIFT**                    #
##                                                             #
##               2. Estimating FHG parameters                  #
##                                                             #
################################################################
##                                                             #
## Version: 1.0                                                #
## Code Author: Kaihao Zheng, Mostaly@pku.edu.cn               #
## Corresponding Author: Peirong Lin, peironglinlin@pku.edu.cn #
##                                                             #
################################################################
##                                                             #
## *Note*:                                                     # 
##   This code is only for technical demonstration.            #
##                                                             #
## Function:                                                   #
## 1. Interative filtering And simplified                      #
##      Logarithmic Regression (LR) on the exponent *b*;       #   
## 2. Parameter Space Sampling (PSS) on parameter *a*;         #
##                                                             #
################################################################


################################################################
## 1. Interative filtering and LR                    
################################################################

# Note: 
#  We set a initial threshold of 20 on floodplain HAND values.
#  For convenient calculation, we first transform reference dataset grids into .csv format, with each row represent a floodplain grid. Locations, upa_river and HAND values are stored.
#  The following codes are conducted on a basin basis.

# initiate packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import floor, ceil
from numba import jit

# initiate paths
csv_path = '' # csv storing reference grid information
upa_file = '' # upa_river files
hnd_file = '' # calculated hand files
ref_file = '' # reference file
upa_tmp_file = '' # *tmp files are regridded to 1km at equator
hnd_tmp_file = ''
ref_tmp_file = ''
df_path = '' # result for parameter b

##########################################
# make csvs from reference datasets

ds1 = gdal.Open(upa_file)
geoTransform = ds1.GetGeoTransform()
minx = geoTransform[0]
maxy = geoTransform[3]
x_tr_1k = 0.008333333300000
y_tr_1k = -0.008333333300000
maxx = minx + geoTransform[1] * ds1.RasterXSize
miny = maxy + geoTransform[5] * ds1.RasterYSize

# bilinear upscaling
os.system(f"gdalwarp -te {minx} {miny} {maxx} {maxy} -r med -tr {x_tr_1k} {y_tr_1k} -crop_to_cutline -overwrite {upa_file} {upa_tmp_file}")
os.system(f"gdalwarp -te {minx} {miny} {maxx} {maxy} -r med -tr {x_tr_1k} {y_tr_1k} -crop_to_cutline -overwrite {ref_file} {ref_tmp_file}")
os.system(f"gdalwarp -te {minx} {miny} {maxx} {maxy} -r med -tr {x_tr_1k} {y_tr_1k} -crop_to_cutline -overwrite {hnd_file} {hnd_tmp_file}")

ds1 = gdal.Open(upa_tmp_file)
dupa = np.array(ds1.GetRasterBand(1).ReadAsArray())
ds2 = gdal.Open(ref_tmp_file)
dref = np.array(ds2.GetRasterBand(1).ReadAsArray())
ds3 = gdal.Open(hnd_tmp_file)
dhnd = np.array(ds3.GetRasterBand(1).ReadAsArray())

# transform into dataframe, select rows and export

df = pd.DataFrame({'mdp':dupa.flat,'ref':dref.flat,'hnd':dhnd.flat})
df1 = df.drop(df.loc[(df['mdp'] == -1.0) | (df['ref'] == 0) | (df['hnd'] == -1) | (df['hnd'] == 0)].index)

df1['ln_upa_a']=np.log((df1['mdp']**(10/3))*1000000)
df1['ln_hnd']=np.log(df1['hnd'])
df1 = df1.sort_values(by='ln_hnd',ascending=True)
df1[['ln_hnd','ln_upa_a']].to_csv(csv_path,index=False,header=True)

################################################
# read reference basins
df_a = pd.read_csv(csv_path)

# designate certain thresholds
baseline = -4.605 # baseline for ln(a0), with a = 0.01 from Nardi et al., 2019
hand_threshold = 20 # we primarily rule out all grids with HAND higher than 20m as it's not plausible.
BinInterval = 100
Thresholds = [5,10,15,20,25,30,35,40,45,50]

# Now mask on Threshold

df_try_a = df_a.loc[(df_a['ref'] != 0) & (df_a['hnd'] != 0)].sort_values('hnd', ascending=True)
df_try_a = df_try_a.loc[(df_try_a['hnd'] < hand_threshold)]
df_try_a['ln_hnd'] = np.log(df_try_a['hnd'])
df_try_a['ln_upa'] = np.log(df_try_a['upa_3']**(10/3))
df_intersection = df_try_a[['hnd','ln_upa','ref']].loc[df_try_a['ref']==3]
df_intersection = df_intersection.rename(columns={'ln_upa':'x','hnd':'y'}) 
df_union = df_try_a[['hnd','ln_upa','ref']].loc[df_try_a['ref']>0]
df_union = df_union.rename(columns={'ln_upa':'x','hnd':'y'}) 
x_min = floor(df_intersection['x'].min())
x_max = ceil(df_intersection['x'].max())

# define necessary functions for filtering, by numba accelaration   

def update_sliding_window_stats(data, x_min, x_max):
    window_stats = []
    for start in np.arange(x_min, x_max, 0.1):
        end = start + 1
        window_df = data[(data[:, 1] >= start) & (data[:, 1] <= end)]
        mean = np.median(window_df[:, 0])
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

# filter!
df_intersection = df_intersection[['y','x']]
data_array = df_intersection.to_numpy()
filtered_data = iterate_filtering(data_array)
df_intersection = pd.DataFrame(filtered_data, columns=['y', 'x'])

################################################
# b sequence and percentile on intersection

result_b_list = []

# sort x into bins
df_intersection['b'] = (df_intersection['y']-baseline)/df_intersection['x'] # assume y = bx + _baseline
df_intersection['x_bin'] = round(df_intersection['x']*BinInterval)/BinInterval
df1 = df_intersection.groupby('x_bin')['b'].agg(max)
list_b = df1.tolist()
list_b = sorted(list_b)

# get percentile
array_b = np.array(list_b)

# save to file
for Threshold in Thresholds:
    b = np.percentile(array_b,100-Threshold)
    result_b_list.append(b)

df_temp = pd.DataFrame({'id':[number]})
for index,i in enumerate(Thresholds):
    df_temp[f'b_{i}']=result_b_list[index]
df_temp.to_csv(df_path)


################################################################
## 2. PSS on a                  
################################################################

# initiate packages
import os
import sys
import glob
from osgeo import gdal
import pandas as pd
import numpy as np

# initiate paths
ref_file = ''
hnd_file = ''
wat_file = ''
upa_file = '' # 
ref_tmp_file = ''
hnd_tmp_file = ''
wat_tmp_file = ''
upa_tmp_file = ''
csv_output = ''

lake_path = '' #
lake_tmp_file = ''

##########################################
# make csvs from reference datasets

# read a file to decide the actual boundary of the 1km*1km basin
ds1 = gdal.Open(hnd_file)
geoTransform = ds1.GetGeoTransform()
minx = geoTransform[0]
maxy = geoTransform[3]
x_tr = geoTransform[1]
y_tr = geoTransform[5]
x_tr_1k = 0.008333333300000
y_tr_1k = -0.008333333300000
maxx = minx + geoTransform[1] * ds1.RasterXSize
miny = maxy + geoTransform[5] * ds1.RasterYSize

# regrid four files into 1km grids
os.system(f"gdalwarp -te {minx} {miny} {maxx} {maxy} -tr {x_tr_1k} {y_tr_1k} -crop_to_cutline -r med -q -overwrite {ref_file} {ref_tmp_file}")
os.system(f"gdalwarp -te {minx} {miny} {maxx} {maxy} -tr {x_tr_1k} {y_tr_1k} -crop_to_cutline -r med -q -overwrite {hnd_file} {hnd_tmp_file}")
os.system(f"gdalwarp -te {minx} {miny} {maxx} {maxy} -tr {x_tr_1k} {y_tr_1k} -crop_to_cutline -r med -q -overwrite {wat_file} {wat_tmp_file}")
os.system(f"gdalwarp -te {minx} {miny} {maxx} {maxy} -tr {x_tr_1k} {y_tr_1k} -crop_to_cutline -r med -q -overwrite {upa_file} {upa_tmp_file}")
os.system(f"gdalwarp -te {minx} {miny} {maxx} {maxy} -tr {x_tr_1k} {y_tr_1k} -crop_to_cutline -r med -q -overwrite {lake_path} {lake_tmp_file}")

# read adjusted reference files
ds1 = gdal.Open(mdp_tmp_file)
dmdp = np.array(ds1.GetRasterBand(1).ReadAsArray())
ds2 = gdal.Open(hnd_tmp_file)
dhnd = np.array(ds2.GetRasterBand(1).ReadAsArray())
ds3 = gdal.Open(wat_tmp_file)
dwat = np.array(ds3.GetRasterBand(1).ReadAsArray())
ds4 = gdal.Open(ref_tmp_file)
dref = np.array(ds4.GetRasterBand(1).ReadAsArray())
ds5 = gdal.Open(lake_tmp_file)
dlake = np.array(ds5.GetRasterBand(1).ReadAsArray())
df = pd.DataFrame({'mdp':dmdp.flat,'hnd':dhnd.flat,'wat':dwat.flat,'ref':dref.flat,'lake':dlake.flat})

# select rows with values not null 
df = df.drop((np.where(df['wat']==-2147483647)[0]))
df.reset_index(drop = True, inplace = True)
df = df.drop((np.where(df['mdp']==-1)[0]))
df.reset_index(drop = True, inplace = True)
df = df.drop((np.where(df['lake']!=0)[0]))

# calculate within dataframe
adjusted_a = 63.095734448
df['upa_3'] = (df['mdp']*adjusted_a)
df = df[['upa_3','hnd','ref']]
df.reset_index(drop = True, inplace = True)

df.to_csv(csv_output, index = False)

################################################
# essential functions for delineations

def Calculate_OA(array1,array0,designated_a):

    count_1 = len(array1)
    count_0 = len(array0)
    count_a = np.count_nonzero(array1 < designated_a)
    count_d = np.count_nonzero(array0 > designated_a)

    if count_1+count_0 == 0:
        return np.nan 
    else:
        return (count_a+count_d)/(count_1+count_0)
    
def Calculate_DOA(array3, array2, array1, array0, designated_a):

    count_3 = len(array3)
    count_2 = len(array2)
    count_1 = len(array1)
    count_0 = len(array0)
    count_total = count_3+count_2+count_1+count_0

    # calculate OA for union
    count_a1 = np.count_nonzero(array1 < designated_a) + np.count_nonzero(array2 < designated_a) + np.count_nonzero(array3 < designated_a)
    count_d1 = np.count_nonzero(array0 > designated_a)

    # calculate OA for intersection
    count_a2 = np.count_nonzero(array3 < designated_a)
    count_d2 = np.count_nonzero(array0 > designated_a) + np.count_nonzero(array1 > designated_a) + np.count_nonzero(array2 > designated_a)

    if count_total == 0:
        return np.nan 
    else:
        return [(count_a1+count_d1)/(count_3+count_2+count_1+count_0), (count_a2+count_d2)/(count_3+count_2+count_1+count_0)]

def Iterate_a(begin, end, array3, array2, array1,array0,k=20):

    # calculate a values in the current interval
    a_values = np.linspace(begin,end,k+1)

    # calculate raw mof values
    DOA_values = [Calculate_DOA(array3, array2,array1,array0,designated_a) for designated_a in a_values] 
    DOA_sum = [(mof[0] + mof[1]) for mof in DOA_values]
    # breakpoint()

    # calculate moving avarage
    moving_avarage=[DOA_sum[0]]
    for i in range(1,len(DOA_sum)-1):
        moving_avarage.append(np.mean(DOA_sum[i-1:i+2]))
    moving_avarage.append(DOA_sum[-1])

    best_DOA = max(moving_avarage)
    a_index = moving_avarage.index(best_DOA)
    best_a = a_values[a_index]

    best_OA_union = DOA_values[a_index][0]
    best_OA_intersection = DOA_values[a_index][1]

    beg = a_values[a_index -1 if a_index > 0 else 0]
    end = a_values[a_index +1 if a_index < k else 0]

    return best_a,best_DOA,(best_OA_union,best_OA_intersection),(beg,end,k)

def Calc_Best_a(df,designated_b,iterate_times = 4):

    start = timeit.default_timer()

    # calculate a_potential from designated b
    df['upa_with_b'] = df['upa_3']**(10/3*designated_b)
    df['a_potential'] = df['hnd']/df['upa_with_b']
    df = df[['a_potential','ref']]
    df.reset_index(drop = True, inplace = True)

    # get result arrays from current DataFrame
    df0 = df[df.ref==0]
    df1 = df[df.ref==1]
    df2 = df[df.ref==2]
    df3 = df[df.ref==3]
    array0 = df0['a_potential'].values # those
    array1 = df1['a_potential'].values
    array2 = df2['a_potential'].values
    array3 = df3['a_potential'].values

    # a self-iterated method to calculate
    best_a = None
    best_mof = None
    begin = 0
    end = 1
    k = 20
    for i in range(iterate_times):
        best_a , best_mof ,(best_OA_union, best_OA_intersection), (begin , end , k) = Iterate_a(begin,end,array3,array2,array1,array0,k=20)

    current = timeit.default_timer()
    print(f'Best a = {best_a:.4f}, With DOA = {best_mof:.6f}, Union = {best_OA_union:.6f}, Intersection = {best_OA_intersection:.6f},. total time = {current-start} seconds.')

    return best_a, best_mof, current - start, best_OA_intersection, best_OA_union


################################################
# PSS iteration

# iterate best a in each 
for index in indices:

    threshold = 30
    b_choice = f'b_{threshold}'

    # read current_b from b_optimize
    current_b = dfb.at[index,b_choice]
    if current_b != current_b:
        continue

    df = pd.read_csv(csv_input)
    df_res = None

    if not os.path.exists(csv_output):
        df_res = pd.DataFrame(columns = ['id','b','a','mof','Union','Intersection'])
    else:
        df_res = pd.read_csv(csv_output)

    for b_choice in b_choices:

        current_b = dfb.at[index,b_choice]
        best_a, best_mof, time, best_OA_intersection, best_OA_union = Calc_Best_a(df,current_b)
        df_temp = pd.DataFrame({'id':[index],'b':[current_b],'a':[best_a],'mof':[best_mof],'Union':[best_OA_union],'Intersection':[best_OA_intersection]})
        df_res = df_res.append(df_temp)

    # breakpoint()
    df_res.to_csv(csv_output, index = False)
    # breakpoint()

# put all csvs together
df_final = pd.DataFrame(columns = ['id','b','a','mof','Union','Intersection'])
for index in indices:

    df_each = pd.read_csv(f'optimize_a/all_csv_v2/no_{index}.csv')
    df_final = df_final.append(df_each)
df_final.to_csv('optimize_a/DOA_v2_fullInfo.csv',index = False)
