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
    G = ox.load_graphml(filepath=path.join(gis_data_path, 'streets', 'cleaning_process_2.graphml'))
    ox.distance.add_edge_lengths(G)

    # delete some dead ends
    nodes_to_remove = [2107289289, 111166066, 297732617, 319898633, 117385463, 293431182, 2273750271,
                      280697142, 309367640, 7867573020, 274610745, 1676206186, 3573081082, 2034593346,
                      33301993, 8307089679, 283734344, 272446352, 5120275681, 2200458285, 199625, 252281626, 295431612,
                      299084684, 3534479782, 1829890434, 130232679, 43511494, 130232678, 378477, 4032457020, 4032457019,
                      4032457015]
    G.remove_nodes_from(nodes_to_remove)

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