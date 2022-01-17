# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 17:00:22 2021

@author: nipat
"""
import osmnx as ox
import numpy as np
import BlueskySCNTools
from plugins.streets.flow_control import street_graph,bbox
from plugins.streets.agent_path_planning import PathPlanning,Path
from plugins.streets.open_airspace_grid import Cell, open_airspace
import os
import dill
import json
import sys
from pympler import asizeof
import math


# Step 1: Import the graph we will be using
dir_path = os.path.dirname(os.path.realpath(__file__))
graph_path = dir_path.replace('current_code', 
          'current_code/whole_vienna/gis/finalized_graph.graphml')
G = ox.io.load_graphml(graph_path)
#G = ox.io.load_graphml('processed_graph.graphml')
edges = ox.graph_to_gdfs(G)[1]
gdf=ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
print('Graph loaded!')



#Load the open airspace grid
input_file=open("renamed_open_airspace_grid.dill", 'rb')
#input_file=open("open_airspace_grid_updated.dill", 'rb')##for 3d path planning
grid=dill.load(input_file)


##Initialise the flow control entity
graph=street_graph(G,edges,grid) 

fig, ax = ox.plot_graph(G,node_color="w",show=False,close=False)


plan = PathPlanning(2,grid,graph,gdf, 16.3228151878,48.2449750623,16.3738585774,48.207272534)
route,turns,edges,next_turn,groups,in_constrained,turn_speed=plan.plan()

x_list=[]
y_list=[]
for r in route:
    x_list.append(r[0])
    y_list.append(r[1])

ax.scatter(x_list,y_list,c="b")


a=(16.3727506,48.2069826)
b=(16.3730102,48.2075894)
c=(16.3738585774,48.207272534)

a=(16.3742024,48.2062878)
b=(16.3740099,48.2064954)
c=(16.3736114,48.2067178)
def angle3pt(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
        Returns a float between 0.0 and 360.0"""
    ang = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

#print(180-angle3pt(a, b, c))

##from geo.py
def rwgs84(latd):
    """ Calculate the earths radius with WGS'84 geoid definition
        In:  lat [deg] (latitude)
        Out: R   [m]   (earth radius) """
    lat    = np.radians(latd)
    a      = 6378137.0       # [m] Major semi-axis WGS-84
    b      = 6356752.314245  # [m] Minor semi-axis WGS-84
    coslat = np.cos(lat)
    sinlat = np.sin(lat)

    an     = a * a * coslat
    bn     = b * b * sinlat
    ad     = a * coslat
    bd     = b * sinlat

    # Calculate radius in meters
    r = np.sqrt((an * an + bn * bn) / (ad * ad + bd * bd))

    return r

##from geo.py
def qdrdist(latd1, lond1, latd2, lond2):
    """ Calculate bearing and distance, using WGS'84
        In:
            latd1,lond1 en latd2, lond2 [deg] :positions 1 & 2
        Out:
            qdr [deg] = heading from 1 to 2
            d [nm]    = distance from 1 to 2 in nm """

    # Haversine with average radius for direction

    # Constants
    nm  = 1852.  # m       1 nautical mile

    # Check for hemisphere crossing,
    # when simple average would not work

    # res1 for same hemisphere
    res1 = rwgs84(0.5 * (latd1 + latd2))

    # res2 :different hemisphere
    a    = 6378137.0       # [m] Major semi-axis WGS-84
    r1   = rwgs84(latd1)
    r2   = rwgs84(latd2)
    res2 = 0.5 * (abs(latd1) * (r1 + a) + abs(latd2) * (r2 + a)) / \
        (np.maximum(0.000001,abs(latd1) + abs(latd2)))

    # Condition
    sw   = (latd1 * latd2 >= 0.)

    r    = sw * res1 + (1 - sw) * res2

    # Convert to radians
    lat1 = np.radians(latd1)
    lon1 = np.radians(lond1)
    lat2 = np.radians(latd2)
    lon2 = np.radians(lond2)


    # Corrected to avoid "nan" at westward direction
    d = r*np.arccos(np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1) + \
                 np.sin(lat1)*np.sin(lat2))

    # Bearing from Ref. http://www.movable-type.co.uk/scripts/latlong.html

    # sin1 = np.sin(0.5 * (lat2 - lat1))
    # sin2 = np.sin(0.5 * (lon2 - lon1))

    coslat1 = np.cos(lat1)
    coslat2 = np.cos(lat2)

    qdr = np.degrees(np.arctan2(np.sin(lon2 - lon1) * coslat2,
                                coslat1 * np.sin(lat2) -
                                np.sin(lat1) * coslat2 * np.cos(lon2 - lon1)))

    return qdr, d/nm
d1=qdrdist(48.2069826,16.3727506,48.2075894,16.3730102)
d2=qdrdist(48.2075894,16.3730102,48.20727253,16.37385858)
print(d1,d2)
plan = PathPlanning(2,grid,graph,gdf, 16.4047372913,48.2025129631,16.3539759332,48.2198151219)
route,turns,edges,next_turn,groups,in_constrained,turn_speed=plan.plan() 
for i in range(1,len(route)-1):
    lon1=route[i-1][0]
    lat1=route[i-1][1]
    lon2=route[i][0]
    lat2=route[i][1]
    lon3=route[i+1][0]
    lat3=route[i+1][1]
    d1=qdrdist(lat1,lon1,lat2,lon2)
    d2=qdrdist(lat2,lon2,lat3,lon3)
    #print(d1,d2)
    dif=abs(d2[0]-d1[0])
    if dif>180:
        dif=360-dif
    print(i,dif)