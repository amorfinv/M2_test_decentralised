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
# use osmnx environment here

'''
Prepare graph for genetic algorithm.

Steps:
    1) Load 'cleaning_process_2.graphml'.
'''

def main():

    # working path
    gis_data_path = 'gis'

    # Load MultiDigraph from create_graph.py
    G = ox.load_graphml(filepath=path.join(gis_data_path, 'streets', 'boundary_clean.graphml'))
    ox.distance.add_edge_lengths(G)

    # convert to gdgs
    nodes, edges = ox.graph_to_gdfs(G)

    # reset directions
    init_edge_directions = graph_funcs.get_first_group_edges(G, edges)
    edges = graph_funcs.set_direction(nodes, edges, init_edge_directions)

    # delete edges smaller than 60 meters
    edges = edges.loc[edges['length'] > 60]

    # get midpoint of edges
    center_points = edges['geometry'].apply(lambda line_geom: line_geom.interpolate(0.5, normalized=True))

    # save to file
    center_points.to_file(path.join(gis_data_path, 'center_points.gpkg'), driver='GPKG')

if __name__ == '__main__':
    main()