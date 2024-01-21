# Readme.md

## Description

This is the documentation of the core codes for SHIFT (Spatial Heterogeneity Improved Floodplain by Terrain Analysis, see reference), a 90-m resolution global geomorphic floodplain map based on terrain analysis. It takes MERIT-Hydro as the terrain input and Floodplain Hydraulic Geometry (FHG) as the thresholding scheme, with the scaling parameters estimated by a stepwise framework that both respects the power law and approximates the spatial extent of hydrodynamic modeling. SHIFT effectively captures the global patterns of the geomorphic floodplains, with better regional details than existing data.

This repository contains the major steps of the development of SHIFT: terrain attribute preparation and parameter estimation. Other details are not included, e.g., pre-processing of terrain inputs, merging or splitting by designated spatial units, the statistical calculation of results, the making of graphs.

We've modified the structure and variable names within the codes and added a few comments for better understanding. The codes are only for demonstration.

## Reference

For data, see: https://zenodo.org/records/10440609

For more methodological and technical details, please check the preprint below:

- Kaihao Zheng, Peirong Lin, Ziyun Yin: SHIFT: A DEM-Based Spatial Heterogeneity Improved Mapping of Global Geomorphic Floodplains. 

Contacts:

- Kaihao Zheng, Mostaly@pku.edu.cn
- Peirong Lin, peironglinlin@pku.edu.cn
