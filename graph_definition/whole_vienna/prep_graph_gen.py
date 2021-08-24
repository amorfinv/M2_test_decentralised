import os
from platform import node
from networkx.classes.function import degree
import osmnx as ox
import geopandas as gpd
import networkx as nx
from osmnx.projection import project_gdf
from osmnx.utils_graph import graph_from_gdfs
from shapely.geometry.point import Point
import graph_funcs
from os import path
import math
import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely import ops
import pandas as pd
import collections
import time
# use osmnx environment here

'''
Prepare graph for genetic algorithm.

Steps:
    1) Load 'cleaning_process_2.graphml'.
    2) get initial directions and use set_directions (direction 0).
    3) save graphml/gpkg of dir_0.
    4) flip initial directions and set directions (direction 1).
    4) save graphm/gpkg of dir_1.
'''

def main():

    # working path
    gis_data_path = 'gis'

    # Load MultiDigraph from cleanup_graph_2.py
    G = ox.load_graphml(filepath=path.join(gis_data_path, 'streets', 'cleaning_process_2.graphml'))

    # convert to gdgs
    nodes, edges = ox.graph_to_gdfs(G)

    # get initial_directions
    init_edge_directions = graph_funcs.get_first_group_edges(G, edges)

    # set direction 0 of edges
    t1 =time.time()
    edges_0 = graph_funcs.set_direction(nodes, edges, init_edge_directions)
    t2 =time.time()
    print(f'graph takes {t2-t1} seconds to direct')

    # create graph and save edited
    G_0 = ox.graph_from_gdfs(nodes, edges_0)

    # save as osmnx graph
    ox.save_graphml(G_0, filepath=path.join(gis_data_path, 'streets', 'graph_dir_0.graphml'))

    # Save geopackage for import to QGIS
    ox.save_graph_geopackage(G_0, filepath=path.join(gis_data_path, 'streets', 'graph_dir_0.gpkg'), directed=True)

    # Now flip direction of graph
    flip_dir = []
    for edge in init_edge_directions:
        flip_dir.append((edge[1],edge[0], 0))

    # get reveresed edges
    t1 =time.time()
    edges_1 = graph_funcs.set_direction(nodes, edges, flip_dir)
    t2 =time.time()
    print(f'graph takes {t2-t1} seconds to direct')
    # create graph and save edited
    G_1 = ox.graph_from_gdfs(nodes, edges_1)   

    # save as osmnx graph
    ox.save_graphml(G_1, filepath=path.join(gis_data_path, 'streets', 'graph_dir_1.graphml'))
    
    # Save geopackage for import to QGIS
    ox.save_graph_geopackage(G_1, filepath=path.join(gis_data_path, 'streets', 'graph_dir_1.gpkg'), directed=True)

if __name__ == '__main__':
    main()