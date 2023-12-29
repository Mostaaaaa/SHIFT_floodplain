# the final step of GFPlain Algorithm, calculates inundation area

# modified by kaihao in 2023/2/5, working in guilt
# now *a* and *b* parameters are both in meters

# kaihao, 2023/11/28
# really have come a long way, but i suppose i will have to share my code later so ...
# rename the whole dataset for the publication of SHIFT

from osgeo import gdal
import numpy as np
import pandas as pd
import argparse
import timeit
import sys

# get parameter from linux

parser = argparse.ArgumentParser(description='need to input two parameters: the pfaf id of the basin, and the file type of the tiles that needs to be merged')
#type是要传入的参数的数据类型  help是该参数的提示信息
parser.add_argument('pfaf_id', type=str, help='pfaf_id')
parser.add_argument('b_value', type=float, help='b_value')
parser.add_argument('a_coef', type=float, help='a_coefficient')
parser.add_argument('name_tag', type=str, help='for name of result fils')
args = parser.parse_args()
start = timeit.default_timer()

def write_geotiff(data,fon):
        # print('... writing to %s ...'%fon)
        ds = ds1
        driver = ds.GetDriver()
        ncols = data.shape[1]
        nrows = data.shape[0]
        dso = driver.Create(fon, ncols, nrows, 1, gdal.GDT_Byte)
        if dso is None:
            print('Could not create output tif')
            sys.exit(1)
        outBand = dso.GetRasterBand(1) #.WriteArray(datao)
        outData = data
        # write the data
        outBand.WriteArray(outData, 0, 0)
        # flush data to disk, set the NoData value and calculate stats
        outBand.FlushCache()
        outBand.SetNoDataValue(255)
        # georeference the image and set the projection
        dso.SetGeoTransform(ds.GetGeoTransform())
        dso.SetProjection(ds.GetProjection())

idd = args.pfaf_id
ttt = f'/home/zhengkh/zkh/floodplain_lev3/no_{idd}'
b_value = args.b_value
a_coef = args.a_coef
# b_tag = f'{b_value}'.split('.')[1]
name_tag = args.name_tag

# #read original data
# print('''-------------------------------
# Performing 6_4_calc_inun.py
# -------------------------------''')
# print('start reading file...')

ds1 = gdal.Open(f'{ttt}/results/max_depth_by_hand_b3.tif')
dmdp = np.array(ds1.GetRasterBand(1).ReadAsArray())
ds2 = gdal.Open(f'{ttt}/results/watershed.tif')
dwat = np.array(ds2.GetRasterBand(1).ReadAsArray())
ds8 = gdal.Open(f'{ttt}/results/hand_by_hand.tif') #watershed raster
dhnd = np.array(ds8.GetRasterBand(1).ReadAsArray())

current = timeit.default_timer()
print(f'finish reading raster files, total time = {current-start} seconds.')

#geospatial info
ncols = ds1.RasterXSize
nrows = ds1.RasterYSize
xoff, a, b, yoff, d, e = ds1.GetGeoTransform()

#create dataframe: EFFICIENT TO USE DATA FRAME FOR ALL DEM ANALYSIS
df = pd.DataFrame({'max_depth':dmdp.flat,'hnd':dhnd.flat,'wat':dwat.flat})
current = timeit.default_timer()
print(f'finish transforming dataframes, total time = {current-start} seconds.')

del ds2,ds8,dmdp,dwat,dhnd

# performing algorithm
adjusted_a = 63.095734448
df['mdp_adjusted'] = (df['max_depth'] * adjusted_a) ** (10/3 * b_value) * a_coef
del df['max_depth']
df['inun'] = (df['hnd']<df['mdp_adjusted']).astype(int)
# breakpoint()
df.loc[df.wat==-2147483647,'inun']=255
current = timeit.default_timer()
print(f'finished calculating inundation, total time = {current-start} seconds.')

#reshape back to 2-D array
inun = np.reshape(df['inun'].values,(nrows,ncols)) # rows first 

#write to GeoTiff
fon = f'{ttt}/results/{name_tag}.tif'
print('... writing to %s ...'%fon)
write_geotiff(inun,fon)
