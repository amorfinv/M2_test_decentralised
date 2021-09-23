from platform import node
import osmnx as ox
import geopandas as gpd
import networkx as nx
import graph_funcs
from os import path
from shapely import ops
import pandas as pd
# use osmnx environment here

'''
'''

def main():

    # working path
    gis_data_path = 'gis'

    # Load MultiDigraph from cleanup_graph_2.py
    G = ox.load_graphml(filepath=path.join(gis_data_path, 'streets', 'hand_edit_parallel.graphml'))

    # convert to gdf
    _, edges = ox.graph_to_gdfs(G)

    # get parallel streets and save to file
    parallel_gdf = graph_funcs.id_parallel_streets(edges, dist_near=32, angle_cut=20)

    # save to file
    parallel_gdf.to_file(path.join(gis_data_path, 'streets', 'parallel_streets_2.gpkg'), driver="GPKG")

if __name__ == '__main__':
    main()