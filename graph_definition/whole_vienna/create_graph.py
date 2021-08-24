import osmnx as ox
import geopandas as gpd
import graph_funcs
from os import path, truncate
import math
import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely import ops
import pandas as pd
import networkx as nx
import collections

'''
Initial processing for graph of Vienna:

Steps:
    1) Download from OSM or load graph from local.
    2) Save graphml and geopackage as "raw_streets*"
    3) Run graph_funcs.initial_prep() to remove some useless attributes.
    4) Remove self loops.
    5) Run graph_funcs.remove_two_way_edges() and graph_funcs.remove_long_way_edges() to 
       convert graph from MultiDiGraph to DiGraph. Note that in name it is still a MultiDigraph
       but it is actually a directed graph.
    6) run graph_funcs.edge_naming_simplification() to remove any edges with key=1.
    7) Save graphml and geopackage as "initial_processing*".
'''
def main():

    # working path
    gis_data_path = 'gis'

    # create MultiDigraph from polygon
    raw_graph_path = path.join(gis_data_path, 'streets', 'raw_streets.graphml')

    # check if graph alrrady downloaded
    if path.exists(raw_graph_path):
        G = ox.load_graphml(raw_graph_path)
    else:
        print('DOWNLOADING!')
        # get airspace polygon from geopackage
        airspace_gdf = gpd.read_file(path.join(gis_data_path, 'airspace', 'constrained_airspace.gpkg', ))
        airspace_gdf.to_crs("EPSG:4326", inplace = True)
        
        airspace_poly = airspace_gdf.geometry.iloc[0]
        G = ox.graph_from_polygon(airspace_poly, network_type='drive', simplify=True, clean_periphery=True, truncate_by_edge=True)
    
        # save as osmnx graph
        ox.save_graphml(G, filepath=raw_graph_path)
        ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'streets', 'raw_streets.gpkg'), directed=True)
    
    # first order of buisness after import of graph is to remove unused attributes from graph
    G = graph_funcs.initial_prep(G)

    # remove self loops from graph
    G.remove_edges_from(nx.selfloop_edges(G))

    # convert graph to node, edge geodataframe
    nodes, edges = ox.graph_to_gdfs(G)
    
    # remove two way edges
    edges = graph_funcs.remove_two_way_edges(edges)
    
    # remove non parallel opposite edges (or long way)
    edges = graph_funcs.remove_long_way_edges(edges)

    # Now fix streets so that all end with key=0
    edges = graph_funcs.edge_naming_simplification(edges)

    # create graph and save edited
    G = ox.graph_from_gdfs(nodes, edges)
    
    # save as osmnx graph
    ox.save_graphml(G, filepath=path.join(gis_data_path, 'streets', 'initial_processing.graphml'))
    
    # Save geopackage for import to QGIS
    ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'streets', 'initial_processing.gpkg'), directed=True)

if __name__ == '__main__':
    main()

