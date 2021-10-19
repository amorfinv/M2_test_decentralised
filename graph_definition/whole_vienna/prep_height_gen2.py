import osmnx as ox
import geopandas as gpd
import graph_funcs
from os import path
import numpy as np
# use osmnx environment here

'''
Prepare graph for height allocation genetic algorithm.
'''

def main():

    # working path
    gis_data_path = 'gis'

    # get airspace polygon from geopackage
    edges = gpd.read_file(path.join(gis_data_path, 'streets', 'prep_height_allocation_new.gpkg'), layer='edges')
    nodes = gpd.read_file(path.join(gis_data_path, 'streets', 'prep_height_allocation_new.gpkg'), layer='nodes')
    
    # recreate edges in format for osmnx
    edges = graph_funcs.edge_gdf_format_from_gpkg(edges)
    nodes = graph_funcs.node_gdf_format_from_gpkg(nodes)

    # add node_degree attribute
    nodes['node_degree'] = graph_funcs.node_degree_attrib(nodes, edges)

    # allocate group heights based on cardinal directions to initialize gen algorithm
    edges['layer_allocation'] = graph_funcs.allocate_group_height(nodes, edges)

    # create graph and save edited
    G = ox.graph_from_gdfs(nodes, edges)

    # add length as attribute
    ox.distance.add_edge_lengths(G)

    # save as graphml
    ox.save_graphml(G, filepath=path.join(gis_data_path, 'streets','prep_height_allocation_2.graphml'))
    
    # Save geopackage for import to QGIS and momepy
    ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'streets','prep_height_allocation_2.gpkg'), directed=True)

if __name__ == '__main__':
    main()