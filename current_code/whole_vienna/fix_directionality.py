import osmnx as ox
import geopandas as gpd
import networkx as nx
from os import path
import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely import ops
import pandas as pd
import graph_funcs
# use osmnx environment here

'''
Fix Directionality from genetic algorithm results. To make it a fully connected graph.
'''

def main():

    # working path
    gis_data_path = 'gis'

    # Load MultiDigraph from create_graph.py
    G = ox.load_graphml(filepath=path.join(gis_data_path, 'layer_heights.graphml'))

    # create a nodes, edges
    nodes, edges = ox.graph_to_gdfs(G)

    # flip the direction of 1342
    edges = graph_funcs.set_direction_group(nodes, edges, (3534558946, 1844621713, 0))

    # flip the direction of 879
    edges = graph_funcs.set_direction_group(nodes, edges, (47998096, 29646666, 0))

    # flip the direction of 1342
    edges = graph_funcs.set_direction_group(nodes, edges, (75569360, 1168873425, 0))

    # convert back to graph
    G = ox.graph_from_gdfs(nodes, edges)

    # ensure directed graph is fully connected
    if nx.is_strongly_connected(G):
        print('Nice job. Graph is strongly connected')

    # save the graph
    ox.save_graphml(G, filepath=path.join(gis_data_path, 'layer_heights_connected.graphml'))

    # save as gpkg
    ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'layer_heights_connected.gpkg'), directed=True)


if __name__ == '__main__':
    main()