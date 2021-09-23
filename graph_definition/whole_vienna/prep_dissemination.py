import osmnx as ox
import geopandas as gpd
import networkx as nx
from shapely.geometry.point import Point
import graph_funcs
from os import path
from shapely import ops
import pandas as pd
import numpy as np

def main():

    # working path
    gis_data_path = 'gis'

    # Load MultiDigraph from boundary_cleanup.py
    G = ox.load_graphml(filepath=path.join(gis_data_path, 'streets', 'boundary_clean.graphml'))

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    # drop some useless columns
    nodes.drop(columns=['street_count'], inplace=True)
    edges.drop(columns=['oneway'], inplace=True)

    nodes['degree'] = graph_funcs.node_degree_attrib(nodes, edges)
    node_osmid_array = nodes.index.to_numpy()

    edge_osmid = []
    for row in edges.iterrows():
        osmid_new = get_new_node_osmid(node_osmid_array)
        node_osmid_array = np.append(node_osmid_array, [osmid_new])
        edge_osmid.append(osmid_new)

    edges['osmid'] = edge_osmid
    G = ox.graph_from_gdfs(nodes, edges)

    # update bearing and length just in case
    G = ox.add_edge_bearings(G)
    G = ox.distance.add_edge_lengths(G)

    # create undirected graph
    G_un = ox.get_undirected(G)

    # save as osmnx graph
    ox.save_graphml(G_un, filepath=path.join(gis_data_path, 'streets', 'cleaned_graphs', 'cleaned.graphml'))

    # Save geopackage for import to QGIS
    ox.save_graph_geopackage(G_un, filepath=path.join(gis_data_path, 'streets', 'cleaned_graphs', 'cleaned.gpkg'))

def get_new_node_osmid(node_osmid_array):
    
    # get max and min node id so we can generate a new node id.
    max_node_id = np.amax(node_osmid_array)
    min_node_id = np.amin(node_osmid_array)

    if min_node_id - 1 > 0: 
        new_node_id = min_node_id - 1
    else:
        new_node_id = max_node_id + 1

    return new_node_id

if __name__ == '__main__':
    main()