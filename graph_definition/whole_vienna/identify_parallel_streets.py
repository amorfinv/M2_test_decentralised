import os
from platform import node
from networkx.classes.function import degree
import osmnx as ox
import geopandas as gpd
from os import path
import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely import ops
import pandas as pd
import collections
from shapely.strtree import STRtree
import rtree

# use osmnx environment here

def main():

    # working path
    gis_data_path = 'gis'

    # Load MultiDigraph from create_graph.py
    G = ox.load_graphml(filepath=path.join(gis_data_path, 'streets', 'regrouping_2.graphml'))

    # convert to gdgs
    nodes, edges = ox.graph_to_gdfs(G)

    # get edge indices
    edges_uv = list(edges.index)
    edge_dict = dict()

    # get list of geometries and add edge attribute as its id
    idx_tree = rtree.index.Index()

    # merged gdf
    edges_gdf= gpd.GeoDataFrame(columns=['u', 'v', 'key', 'geometry'], crs=edges.crs)
    edges_gdf.set_index(['u', 'v', 'key'], inplace=True)


    i = 0
    for index, row in edges.iterrows():
        
        geom = row.loc['geometry']
        edge_dict[i] = index
        idx_tree.insert(i, geom.bounds)

        i += 1

    closest_dict = dict()
    for index_edge, row in edges.iterrows():
        geom = row.loc['geometry']
        group1 = row.loc['stroke_group']
        nearest = []
        geom_merge = []
        geom_merge.append(geom)
        lats1, lons1 = geom.xy
        l1 = [[lats1[0], lons1[0]], [lats1[-1], lons1[-1]]]

        j = 1
        while True:

            nearest_trial = list(idx_tree.nearest(geom.bounds, j))
            temp_edge_id = edge_dict[nearest_trial[-1]]
            temp_geom = edges.loc[temp_edge_id, 'geometry']
            group2 = edges.loc[temp_edge_id, 'stroke_group']

            j += 1

            # check intersection
            if temp_geom.intersects(geom):
                continue
            
            # check bearing
            lats2, lons2 = temp_geom.xy
            l2 = [[lats2[0], lons2[0]], [lats2[-1], lons2[-1]]]
            between_angle = angle(l1, l2)
            
            if between_angle > 20:
                continue
            
            if group1 == group2:
                continue
            
            # get nearest points into geometry and get distance in meters
            p1, p2 = ops.nearest_points(geom, temp_geom)
            dist = qdrdist(p1.y, p1.x, p2.y, p2.x)

            if dist <= 32:
                nearest.append(temp_edge_id)
                geom_merge.append(temp_geom)
            else:
                break
        
        # only append if nearest exists 
        if nearest:
            closest_dict[index_edge] = nearest
            collection_lines = MultiLineString(geom_merge)

            # create geodatarame
            row_dict = dict()
            row_dict['u'] = index_edge[0]
            row_dict['v'] = index_edge[1]
            row_dict['key'] = index_edge[2]
            row_dict['geometry'] = [collection_lines]

            edge_gdf_new = gpd.GeoDataFrame(row_dict, crs=edges.crs)
            edge_gdf_new.set_index(['u', 'v', 'key'], inplace=True)
            edges_gdf = edges_gdf.append(edge_gdf_new)    


    # write to file
    edges_gdf.to_file(path.join(gis_data_path, 'streets', 'parallel_streets.gpkg'), driver="GPKG")




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

    # sin1 = np.sin(0.5 * (lat2 - lat1))
    # sin2 = np.sin(0.5 * (lon2 - lon1))

    # coslat1 = np.cos(lat1)
    # coslat2 = np.cos(lat2)


    # qdr = np.degrees(np.arctan2(np.sin(lon2 - lon1) * coslat2,
    #     coslat1 * np.sin(lat2) - np.sin(lat1) * coslat2 * np.cos(lon2 - lon1)))

    return d


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

def angle(l1,l2):
    l1 = np.array(l1)
    l2 = np.array(l2)
    m1 = (l1[1,1]-l1[0,1])/(l1[1,0]-l1[0,0])
    m2 = (l2[1,1]-l2[0,1])/(l2[1,0]-l2[0,0])
    angle_rad = abs(np.arctan(m1) - np.arctan(m2))
    angle_deg = angle_rad*180/np.pi
    return angle_deg

if __name__ == '__main__':
    main()