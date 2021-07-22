# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:00:22 2021

@author: andub
"""
import numpy as np
from shapely.geometry import LineString
from shapely.strtree import STRtree
from shapely.ops import nearest_points
from rtree import index
import copy
nm  = 1852.


def calcDirectionality(group_gdf, nodes_gdf, edge_directions):
    # We need to create an rtree for each direction
    # Fow now, let's consider two directions
    treeNS = index.Index()
    treeEW = index.Index()
    geomsNS = []
    geomsEW = []
    bool_list = [None] * len(edge_directions)
    directions = copy.copy(edge_directions)
    for idx, group in enumerate(group_gdf.geometry):
        sort_var = sortLine(group)
        if sort_var == 'NS':
            group.id = idx
            treeNS.insert(idx, group.bounds)
            geomsNS.append(group)
        elif sort_var == 'EW':
            group.id = idx
            treeEW.insert(idx, group.bounds)
            geomsEW.append(group)

    for group in geomsNS:
        neighbors = getNeighbors(group, geomsNS)
        print(group.id, neighbors)
        
    for group in geomsEW:
        neighbors = getNeighbors(group, geomsEW)
        print(group.id, neighbors)
        
    for i in range(len(bool_list)):
        if bool_list[i] == 1 or bool_list[i] == True:
            direction = copy.copy(directions[i])
            directions[i] = (direction[1], direction[0], direction[2])
            
    return edge_directions

def getNeighbors(line, geoms, num = 10):
    '''Returns <num> closest neighbors of line that are within twice the distance
    of the second neighbor.'''
    neighbors = []
    distances = []
    # First, remove the line we're looking for
    geomlist = copy.copy(geoms)
    geomlist.pop(geomlist.index(line))
    # Get neighbors
    for i in range(num):
        # Return if templist empty
        if len(geomlist) == 0:
            return neighbors
        # Create query-only strtree
        strtreeNS = STRtree(geomlist)
        neighbor = strtreeNS.nearest(line)
        p1, p2 = nearest_points(neighbor, line)
        qdr, dist = qdrdist(p1.y, p1.x, p2.y, p2.x)
        distances.append(dist * nm)
        
        # Check for distances
        if i>1 and distances[-1] > distances[1] * 2:
            return neighbors
        
        neighbors.append(neighbor.id)
        
        # Eliminate from geom list
        geomlist.pop(geomlist.index(neighbor))
    return neighbors

def sortLine(line):
    '''Sorts a line by bearing, returns a string.'''
    # Extract max and min coordinates of line. Bounds are xmin, ymin, xmax, ymax.
    lon1, lat1, lon2, lat2 = line.bounds
    qdr, dist = qdrdist(lat1, lon1, lat2, lon2)
    
    if dist * 1852 < 200:
        return None
    
    # QDR is between -180 and 180 deg
    if (qdr < -135) or (135 < qdr) or (-45 < qdr < 45):
        return 'NS'
    else:
        return 'EW'

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

def qdrdist(latd1, lond1, latd2, lond2):
    """ Calculate bearing and distance, using WGS'84
        In:
            latd1,lond1 en latd2, lond2 [deg] :positions 1 & 2
        Out:
            qdr [deg] = heading from 1 to 2
            d [nm]    = distance from 1 to 2 in nm """

    # Haversine with average radius for direction

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

    
    #root = sin1 * sin1 + coslat1 * coslat2 * sin2 * sin2
    #d    =  2.0 * r * np.arctan2(np.sqrt(root) , np.sqrt(1.0 - root))
    # d =2.*r*np.arcsin(np.sqrt(sin1*sin1 + coslat1*coslat2*sin2*sin2))

    # Corrected to avoid "nan" at westward direction
    d = r*np.arccos(np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1) + \
                 np.sin(lat1)*np.sin(lat2))

    # Bearing from Ref. http://www.movable-type.co.uk/scripts/latlong.html

    sin1 = np.sin(0.5 * (lat2 - lat1))
    sin2 = np.sin(0.5 * (lon2 - lon1))

    coslat1 = np.cos(lat1)
    coslat2 = np.cos(lat2)


    qdr = np.degrees(np.arctan2(np.sin(lon2 - lon1) * coslat2,
        coslat1 * np.sin(lat2) - np.sin(lat1) * coslat2 * np.cos(lon2 - lon1)))

    return qdr, d/nm
