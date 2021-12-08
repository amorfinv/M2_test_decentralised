from unicodedata import name
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString, Polygon, LinearRing
import graph_funcs
from os import path
import numpy as np
import pandas as pd
from pympler import asizeof

"""
Code renames the graph nodes and edges ids to be consecutive integers starting from 0.

THIS IS THE FINAL GRAPH
"""

def main():
    # working path
    gis_data_path = 'gis'

    # Load MultiDigraph from create_graph.py
    G = ox.load_graphml(filepath=path.join(gis_data_path, 'layer_heights_simplified_2.graphml'))

    # convert to gdgs
    nodes, edges = ox.graph_to_gdfs(G)
    
    # rename node indices
    nodes, edges = rename_ids(nodes, edges)

    # convert back to graph and save
    G = ox.graph_from_gdfs(nodes, edges)

    ox.save_graphml(G, filepath=path.join(gis_data_path, 'renamed_graph.graphml'))

    ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'renamed_graph.gpkg'), directed=True)

def rename_ids(nodes, edges):
    
    # make a copy
    nodes_gdf = nodes.copy()
    edges_gdf = edges.copy()

    # FIRST DO THE NODES
    # get the old indices
    old_node_indices = nodes_gdf.index

    # now make a numpy array of new ordered indices
    new_node_indices = np.arange(len(old_node_indices))

    # make a dictionary of old node indices and new indices
    old_to_new_nodes = dict(zip(old_node_indices, new_node_indices))

    # drop the old index
    nodes_gdf.reset_index(drop=True, inplace=True)

    # set new_indices as index of gdf
    nodes_gdf.set_index(new_node_indices, inplace=True)

    # NOW rename the edges
    # get the old indices
    old_edge_indices = edges_gdf.index.to_list()

    # loop through the old_edge_indices and get new names with old_to_new_nodes
    u_v_k = [(old_to_new_nodes[old_index[0]], old_to_new_nodes[old_index[1]], 0) for old_index in old_edge_indices]
    u_v = np.array([(old_to_new_nodes[old_index[0]], old_to_new_nodes[old_index[1]]) for old_index in old_edge_indices], dtype='int16,int16')
    
    # drop the indices of the gdf
    edges_gdf.reset_index(drop=True, inplace=True)

    # make a multiindex
    new_index = pd.MultiIndex.from_tuples(u_v_k, names=['u', 'v', 'key'])

    # set the new index for edges_gdf
    edges_gdf.set_index(new_index, inplace=True)
    
    return nodes_gdf, edges_gdf

if __name__ == '__main__':
    main()
