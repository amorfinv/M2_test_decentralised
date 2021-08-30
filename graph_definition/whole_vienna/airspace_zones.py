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
Do another clean of graph using hand edits from (hand_edtis_2.gpkg).

Steps:
    1) Load 'hand_edits_2.gpkg' as an omsnx style gdf.
    2) Do some more cleaning up edits to clean up some intersections. Connect some nodes, remove edges ..etc
    3) Run COINS and set initial direction of edges.
    4) Simplify graph by removing degree-2 nodes with large interior angle and run COINS.
    8) Save graph and gdf as 'cleaning_process_2.*'
'''

def main():

    # manually edit graph prior to doing stuff here

    # working path
    gis_data_path = 'gis'

    # get airspace polygon from geopackage
    edges = gpd.read_file(path.join(gis_data_path, 'streets', 'sub_groups','zone_a.gpkg'), layer='edges')
    nodes = gpd.read_file(path.join(gis_data_path, 'streets', 'sub_groups','zone_a.gpkg'), layer='nodes')
    
    # recreate edges in format for osmnx
    edges_a = graph_funcs.edge_gdf_format_from_gpkg(edges)
    nodes_a = graph_funcs.node_gdf_format_from_gpkg(nodes)

    # create graph and save edited
    G_a = ox.graph_from_gdfs(nodes_a, edges_a)

    print('--------FINISHED ZONE A---------')

    # get airspace polygon from geopackage
    edges = gpd.read_file(path.join(gis_data_path, 'streets', 'sub_groups','zone_b.gpkg'), layer='edges')
    nodes = gpd.read_file(path.join(gis_data_path, 'streets', 'sub_groups','zone_b.gpkg'), layer='nodes')
    
    # recreate edges in format for osmnx
    edges_b = graph_funcs.edge_gdf_format_from_gpkg(edges)
    nodes_b = graph_funcs.node_gdf_format_from_gpkg(nodes)

    # create graph and save edited
    G_b = ox.graph_from_gdfs(nodes_b, edges_b)

    print('--------FINISHED ZONE B---------')

    # get airspace polygon from geopackage
    nodes = gpd.read_file(path.join(gis_data_path, 'streets', 'sub_groups','zone_c.gpkg'), layer='edges')
    edges = gpd.read_file(path.join(gis_data_path, 'streets', 'sub_groups','zone_c.gpkg'), layer='nodes')
    
    # recreate edges in format for osmnx
    edges_c = graph_funcs.edge_gdf_format_from_gpkg(edges)
    nodes_c = graph_funcs.node_gdf_format_from_gpkg(nodes)

    # create graph and save edited
    G_c = ox.graph_from_gdfs(nodes_c, edges_c)

    print('--------FINISHED ZONE C---------')

    # get airspace polygon from geopackage
    edges = gpd.read_file(path.join(gis_data_path, 'streets', 'sub_groups','zone_d.gpkg'), layer='edges')
    nodes = gpd.read_file(path.join(gis_data_path, 'streets', 'sub_groups','zone_d.gpkg'), layer='nodes')
    
    # recreate edges in format for osmnx
    edges_d = graph_funcs.edge_gdf_format_from_gpkg(edges)
    nodes_d = graph_funcs.node_gdf_format_from_gpkg(nodes)

    # create graph and save edited
    G_d = ox.graph_from_gdfs(nodes_d, edges_d)

    print('--------FINISHED ZONE D---------')

    # get airspace polygon from geopackage
    edges = gpd.read_file(path.join(gis_data_path, 'streets', 'sub_groups','zone_e.gpkg'), layer='edges')
    nodes = gpd.read_file(path.join(gis_data_path, 'streets', 'sub_groups','zone_e.gpkg'), layer='nodes')
    
    # recreate edges in format for osmnx
    edges_e = graph_funcs.edge_gdf_format_from_gpkg(edges)
    nodes_e = graph_funcs.node_gdf_format_from_gpkg(nodes)

    # create graph and save edited
    G_e = ox.graph_from_gdfs(nodes_e, edges_e)

    print('--------FINISHED ZONE E---------')

    # get airspace polygon from geopackage
    edges = gpd.read_file(path.join(gis_data_path, 'streets', 'sub_groups','zone_f.gpkg'), layer='edges')
    nodes = gpd.read_file(path.join(gis_data_path, 'streets', 'sub_groups','zone_f.gpkg'), layer='nodes')
    
    # recreate edges in format for osmnx
    edges_f = graph_funcs.edge_gdf_format_from_gpkg(edges)
    nodes_f = graph_funcs.node_gdf_format_from_gpkg(nodes)

    # create graph and save edited
    G_f = ox.graph_from_gdfs(nodes_e, edges_e)

    print('--------FINISHED ZONE F---------')

    # # save as osmnx graph
    # ox.save_graphml(G, filepath=path.join(gis_data_path, 'streets', 'cleaning_process_2.graphml'))
    
    # # # # Save geopackage for import to QGIS and momepy
    # ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'streets', 'cleaning_process_2.gpkg'), directed=True)


if __name__ == '__main__':
    main()