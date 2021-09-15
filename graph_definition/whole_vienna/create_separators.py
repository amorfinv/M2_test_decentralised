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
Make separators

Steps:
'''

def main():

    # working path
    gis_data_path = 'gis'

    # Load MultiDigraph from create_graph.py
    G = ox.load_graphml(filepath=path.join(gis_data_path, 'streets', 'regrouping_3.graphml'))

    # convert to gdgs
    nodes, edges = ox.graph_to_gdfs(G)

    # ##### regrouping # 1
    # regroup 1a: split groups at (big highway) starting (48.23460146649045, 16.354987971444462)
    edge_a = (199683, 307865329, 0)
    edge_b = (307865329, 33344230, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)

    # regroup 1b at node 451666739
    edge_a = (378462, 451666739, 0)
    edge_b = (378464, 451666739, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)

    edge_direction = (1170707214, 1170706775, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)

    ##### regrouping # 2
    # regroup 2a: split groups at (big highway starting at 48.246105506044714, 16.381461118852457) node 24950487
    edge_a = (199684, 24950487, 0)
    edge_b = (24950487, 24950483, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)

    # regroup 2b at node 24950483
    edge_a = (24950487, 24950483, 0)
    edge_b = (24950483, 1371105029, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)

    # split group at a node
    node_split = 281224229
    edges = graph_funcs.split_group_at_node(edges, node_split, '26')

    # regroup 2c at node 281224229
    edge_a = (27379233, 281224229, 0)
    edge_b = (34767011, 281224229, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)

    edge_direction = (24967175, 395058, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)
    
    # regroup 2d at node 34767011
    edge_a = (281224229, 34767011, 0)
    edge_b = (34767011, 34767150, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)

    ##### regrouping #5
    # regrouping #5a node 319836124
    edge_a = (319836124, 935742456, 0)
    edge_b = (319836124, 27026891, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)

    edges = graph_funcs.split_group_at_node(edges, 1212970742, '812')

    #### regrouiping # 6
    edges = graph_funcs.split_group_at_node(edges, 78374656, '441')

    # regrouping #6a node 311045196
    edge_a = (311045196, 685148, 0)
    edge_b = (394359, 311045196, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)

    #### regrouiping # 7
    edges = graph_funcs.split_group_at_node(edges, 685048, '0')

    # create graph and save edited
    G = ox.graph_from_gdfs(nodes, edges)

    # save as osmnx graph
    ox.save_graphml(G, filepath=path.join(gis_data_path, 'streets', 'regrouping_4.graphml'))

    # Save geopackage for import to QGIS a
    ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'streets', 'regrouping_4.gpkg'), directed=True)

if __name__ == '__main__':
    main()