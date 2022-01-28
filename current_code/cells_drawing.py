# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 10:37:54 2022

@author: nipat
"""

import geopandas
import shapely.geometry
import matplotlib.pyplot as plt


import matplotlib.patches as patches
from PIL import Image
import numpy as np

from matplotlib.patches import Polygon
import math
import copy


from plugins.streets.open_airspace_grid import Cell, open_airspace
import dill
from pyproj import  Transformer


#https://geopandas.org/docs/reference/api/geopandas.GeoDataFrame.to_crs.html
countries_gdf = geopandas.read_file("geofences_big.gpkg")
city_gdf = geopandas.read_file("updated_constrained_airspace.gpkg")

fig, ax = plt.subplots(1,1)



city_poly=list(city_gdf["geometry"][0].exterior._get_coords())
del(city_poly[-1])

# =============================================================================
# for i in city_poly:
#     ax.scatter(i[0],i[1],color='g')
# 
# =============================================================================
city_poly.reverse()

##Plots

plt.ylim([5330000,5350000])
plt.xlim([593000,609000])

plt.ylim([5335000,5337500])
plt.xlim([595000,600000])


# =============================================================================
# plt.ylim([5340000,5346000])
# plt.xlim([595000,598000])
# =============================================================================

countries_gdf.plot(ax=ax)

#countries_gdf.loc[[24],'geometry'].plot(ax=ax)

city_gdf.plot(ax=ax,facecolor="k")

#Load the open airspace grid
input_file=open("renamed_open_airspace_grid.dill", 'rb')
input_file=open("open_airspace_final.dill", 'rb')
#input_file=open("open_airspace_grid_updated.dill", 'rb')##for 3d path planning
grid=dill.load(input_file)



c=['g','r','y','k']
negh_list=[]
for i in range(224,len(grid.grid)):
    p=grid.grid[i]
    for j in p.neighbors:
        
        negh_list.append(j)
gr_k_in=[]       
for i in range(len(grid.grid)):
#for i in range(1):
    
    p=grid.grid[i]
    gr_k_in.append(p.key_index)
    y = np.array([[p.p0[0], p.p0[1]], [p.p1[0], p.p1[1]], [p.p2[0] ,p.p2[1]], [p.p3[0], p.p3[1]]])
    #p = Polygon(y, facecolor = c[i%4])
    pol = Polygon(y, facecolor = "none",edgecolor="r")

    if i==4482-4481:
        pol = Polygon(y, facecolor = "none",edgecolor="g")
    if i==4522-4481:
        pol = Polygon(y, facecolor = "none",edgecolor="y")      

    ax.add_patch(pol)  

transformer = Transformer.from_crs('epsg:4326','epsg:32633')
p=transformer.transform(48.1739801195,16.3050401289)
ax.scatter(p[0],p[1],c="y")
print(p)
ax.scatter(597715.1864200250711292 ,5336619.49283800181001425,c="y")
