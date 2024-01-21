################################################################
################################################################
##                                                             #
##                 **Core Codes for SHIFT**                    #
##                                                             #
##             1. Calculating Terrain Attributes               #
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
## 1. Calculate topography for pixels by flow direction,       #
##    for HAND and UPA_river calculation.                      #
## 2. Calculate HAND (by meter) by flow direction.             #   \
##                                                             #
################################################################


import os
import glob
import argparse


################################################################
## 1. Calculate topography for pixels by flow direction        #
################################################################

# Note: 
#  We reinvent the wheels for HAND calculation because we've encountered certain troubles when applying TauDEM toolkit or whitebox for HAND. Nevertheless these two are obviously more stable realization of HAND delineation, so if you can prepare HAND and UPA_river by other means you can skip this part.
#  The "topography" defined here means how each land grid "points to" the next grid by flow direction.

# initiate packages
from osgeo import gdal
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point,LineString,Polygon
import argparse
import timeit
import sys
import os
import pickle # for dumping temporary topography files.
pickle_protocol=4
sys.setrecursionlimit(200000) # otherwise may exceed

# initiate paths
river_network_path = '' # string delineated by 1000km2 threshold
watershed_path = '' # watershed delineated by 1000km2 threshold
direction_path = '' # direction file by TauDEM format, 1: east, 8: southeast, 7: south, 6: southwest, 5: west, 4: northwest, 3: north. 2: northeast, 0: river mouth, inland depression, undefined (ocean)
topo_path = '' # output topography paths

# read original data
ds1 = gdal.Open(river_network_path) 
dstr = np.array(ds1.GetRasterBand(1).ReadAsArray())
ds2 = gdal.Open(watershed_path) 
dwat = np.array(ds2.GetRasterBand(1).ReadAsArray())
ds3 = gdal.Open(f'{ttt}/crop_files/dir_modify.tif') 
ddir = np.array(direction_path)

# get geospatial info
ncols = ds1.RasterXSize
nrows = ds1.RasterYSize

# create dataframe
df = pd.DataFrame({'riv':dstr.flat,'dir':ddir.flat,'wat':dwat.flat})
del ds1,ds3,dstr,ddir,dwat,ds2 # delete for space saving

# set indices in df.
# failed creating index (x,y) to every pixel: memory exceeded
# mapping image indices (x,y) to df indices i: i = 24000 * y + x
# mapping df indices i to image indices (x,y): y = i // 24000, x = i % 24000
df['flowto_ix'] = df.index % ncols
df['flowto_iy'] = df.index // ncols

# initiating HAND values for 
df['hnd'] = -1
df.loc[(df.riv == 1),'hnd'] = 0
current = timeit.default_timer()
del df['riv']

# trace next vertex
k = df['dir'].isin([1,2,8])  #to east
df.loc[k,'flowto_ix'] = df['flowto_ix'][k]+1
k = df['dir'].isin([6,5,4])  #to west
df.loc[k,'flowto_ix'] = df['flowto_ix'][k]-1
k = df['dir'].isin([4,3,2]) #to north
df.loc[k,'flowto_iy'] = df['flowto_iy'][k]-1
k = df['dir'].isin([8,7,6])  #to south
df.loc[k,'flowto_iy'] = df['flowto_iy'][k]+1 
df['next_i'] = ncols * df['flowto_iy'] + df['flowto_ix']
del df['flowto_ix'],df['flowto_iy']

# set null values
k = df['dir'] == 0
df.loc[k,'next_i'] = -1
df['next_i'] = df['next_i'].astype(np.int32)
del df['dir']

# save fliles
output = df.to_pickle(topo_path)


################################################################
## 2. Calculate HAND (by meter)                                #
################################################################

# Note:
#  From our experience, MERIT-Hydro data are somewhat incompatible with existing toolkits.
#  upa_river is calculated in a similar manner.

# initiate packages
from numba import jit

# initiate paths
elevation_path = '' # for elevation file
HAND_path = '' # for HAND output

# define a output function for writing .tif file
def write_geotiff(ds,data,fon):
        driver = ds.GetDriver()
        ncols = data.shape[1]
        nrows = data.shape[0]
        dso = driver.Create(fon, ncols, nrows, 1, gdal.GDT_Float32)
        if dso is None:
            print('Could not create output tif')
            sys.exit(1)
        outBand = dso.GetRasterBand(1) #.WriteArray(datao)
        outData = data
        # write the data
        outBand.WriteArray(outData, 0, 0)
        # flush data to disk, set the NoData value and calculate stats
        outBand.FlushCache()
        outBand.SetNoDataValue(-1)
        # georeference the image and set the projection
        dso.SetGeoTransform(ds.GetGeoTransform())
        dso.SetProjection(ds.GetProjection())

# read files
ds1 = gdal.Open(elevation_path)
ddem = np.array(ds1.GetRasterBand(1).ReadAsArray())
df = pd.read_pickle(topo_path)

#geospatial info
ncols = ds1.RasterXSize
nrows = ds1.RasterYSize
xoff, a, b, yoff, d, e = ds1.GetGeoTransform()

# pre-processing
df_wat_null_value = df.iloc[0]['wat'] # null values that are not located within a watershed.
kk = (df['wat'] != df_wat_null_value) & (df['hnd'] < 0) & (df['next_i'] > 0) # pixels needed recursion
Valid_grids = np.where(kk == True)
kk_count = len(Valid_grids[0])
kk_unit = kk_count//10000
curr_count = 0

# transform dataset to numpy array so as to utilize numba acceleration
hnd_array = df['hnd'].to_numpy().astype(np.float32)
dem_array = np.array(ddem.flat)
nxt_array = df['next_i'].to_numpy()
del df

@jit(nopython=True) # using numba to accelerate
def CalcNext(i, hnd_array , dem_array , nxt_array):
    # a function that traces the height in DEM adjacent DEM

    # tails
    if hnd_array[i] >= 0:
        return hnd_array[i]

    # recursives
    _next_i = nxt_array[i]
    if _next_i == -1:
        return 0

    next_hnd = CalcNext(_next_i, hnd_array , dem_array , nxt_array)
    d_height = dem_array[i] - dem_array[_next_i]
    hnd_array[i] = next_hnd + d_height

    return next_hnd + d_height
    
# start tracing
for j in Valid_grids[0]:
    CalcNext(j, hnd_array , dem_array , nxt_array)
    
#reshape back to 2-D array
hnd = np.reshape(hnd_array ,(nrows,ncols)) # rows first 

#write to GeoTiff
write_geotiff(hnd,HAND_path)


