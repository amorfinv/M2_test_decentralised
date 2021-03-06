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

    # add layer allocation
    edges_a['layer_allocation'] = graph_funcs.allocate_group_height(nodes_a, edges_a)

    # calculate integral bearing difference
    edges_geometry = edges_a['geometry'].to_numpy()
    edges_a['integral_bearing_diff'] = graph_funcs.calculate_integral_bearing_difference(edges_geometry)

    # create graph and save edited
    G_a = ox.graph_from_gdfs(nodes_a, edges_a)

    # save as graphml
    ox.save_graphml(G_a, filepath=path.join(gis_data_path, 'streets', 'sub_groups','zone_a_gen.graphml'))
    
    # Save geopackage for import to QGIS and momepy
    ox.save_graph_geopackage(G_a, filepath=path.join(gis_data_path, 'streets', 'sub_groups','zone_a_gen.gpkg'), directed=True)

    print('--------FINISHED ZONE A---------')

    # get airspace polygon from geopackage
    edges = gpd.read_file(path.join(gis_data_path, 'streets', 'sub_groups','zone_b.gpkg'), layer='edges')
    nodes = gpd.read_file(path.join(gis_data_path, 'streets', 'sub_groups','zone_b.gpkg'), layer='nodes')
    
    # recreate edges in format for osmnx
    edges_b = graph_funcs.edge_gdf_format_from_gpkg(edges)
    nodes_b = graph_funcs.node_gdf_format_from_gpkg(nodes)

    # add layer allocation
    edges_b['layer_allocation'] = graph_funcs.allocate_group_height(nodes_b, edges_b, rotation_val=20)

    # calculate integral bearing difference
    edges_geometry = edges_b['geometry'].to_numpy()
    edges_b['integral_bearing_diff'] = graph_funcs.calculate_integral_bearing_difference(edges_geometry)

    # create graph and save edited
    G_b = ox.graph_from_gdfs(nodes_b, edges_b)

    # save as graphml
    ox.save_graphml(G_b, filepath=path.join(gis_data_path, 'streets', 'sub_groups','zone_b_gen.graphml'))
    
    # Save geopackage for import to QGIS and momepy
    ox.save_graph_geopackage(G_b, filepath=path.join(gis_data_path, 'streets', 'sub_groups','zone_b_gen.gpkg'), directed=True)

    print('--------FINISHED ZONE B---------')

    # get airspace polygon from geopackage
    nodes = gpd.read_file(path.join(gis_data_path, 'streets', 'sub_groups','zone_c.gpkg'), layer='edges')
    edges = gpd.read_file(path.join(gis_data_path, 'streets', 'sub_groups','zone_c.gpkg'), layer='nodes')
    
    # recreate edges in format for osmnx
    edges_c = graph_funcs.edge_gdf_format_from_gpkg(edges)
    nodes_c = graph_funcs.node_gdf_format_from_gpkg(nodes)

    # add layer allocation
    edges_c['layer_allocation'] = graph_funcs.allocate_group_height(nodes_c, edges_c, rotation_val=45)

    # calculate integral bearing difference
    edges_geometry = edges_c['geometry'].to_numpy()
    edges_c['integral_bearing_diff'] = graph_funcs.calculate_integral_bearing_difference(edges_geometry)

    # create graph and save edited
    G_c = ox.graph_from_gdfs(nodes_c, edges_c)

    # save as graphml
    ox.save_graphml(G_c, filepath=path.join(gis_data_path, 'streets', 'sub_groups','zone_c_gen.graphml'))
    
    # Save geopackage for import to QGIS and momepy
    ox.save_graph_geopackage(G_c, filepath=path.join(gis_data_path, 'streets', 'sub_groups','zone_c_gen.gpkg'), directed=True)

    print('--------FINISHED ZONE C---------')

    # get airspace polygon from geopackage
    edges = gpd.read_file(path.join(gis_data_path, 'streets', 'sub_groups','zone_d.gpkg'), layer='edges')
    nodes = gpd.read_file(path.join(gis_data_path, 'streets', 'sub_groups','zone_d.gpkg'), layer='nodes')
    
    # recreate edges in format for osmnx
    edges_d = graph_funcs.edge_gdf_format_from_gpkg(edges)
    nodes_d = graph_funcs.node_gdf_format_from_gpkg(nodes)

    # add layer allocation
    edges_d['layer_allocation'] = graph_funcs.allocate_group_height(nodes_d, edges_d, rotation_val=-45)

    # calculate integral bearing difference
    edges_geometry = edges_d['geometry'].to_numpy()
    edges_d['integral_bearing_diff'] = graph_funcs.calculate_integral_bearing_difference(edges_geometry)

    # create graph and save edited
    G_d = ox.graph_from_gdfs(nodes_d, edges_d)

    # save as graphml
    ox.save_graphml(G_d, filepath=path.join(gis_data_path, 'streets', 'sub_groups','zone_d_gen.graphml'))
    
    # Save geopackage for import to QGIS and momepy
    ox.save_graph_geopackage(G_d, filepath=path.join(gis_data_path, 'streets', 'sub_groups','zone_d_gen.gpkg'), directed=True)

    print('--------FINISHED ZONE D---------')

    # get airspace polygon from geopackage
    edges = gpd.read_file(path.join(gis_data_path, 'streets', 'sub_groups','zone_e.gpkg'), layer='edges')
    nodes = gpd.read_file(path.join(gis_data_path, 'streets', 'sub_groups','zone_e.gpkg'), layer='nodes')
    
    # recreate edges in format for osmnx
    edges_e = graph_funcs.edge_gdf_format_from_gpkg(edges)
    nodes_e = graph_funcs.node_gdf_format_from_gpkg(nodes)

    # add layer allocation
    edges_e['layer_allocation'] = graph_funcs.allocate_group_height(nodes_e, edges_e, rotation_val=45)

    # calculate integral bearing difference
    edges_geometry = edges_e['geometry'].to_numpy()
    edges_e['integral_bearing_diff'] = graph_funcs.calculate_integral_bearing_difference(edges_geometry)

    # create graph and save edited
    G_e = ox.graph_from_gdfs(nodes_e, edges_e)

    # save as graphml
    ox.save_graphml(G_e, filepath=path.join(gis_data_path, 'streets', 'sub_groups','zone_e_gen.graphml'))
    
    # Save geopackage for import to QGIS and momepy
    ox.save_graph_geopackage(G_e, filepath=path.join(gis_data_path, 'streets', 'sub_groups','zone_e_gen.gpkg'), directed=True)

    print('--------FINISHED ZONE E---------')

    # get airspace polygon from geopackage
    edges = gpd.read_file(path.join(gis_data_path, 'streets', 'sub_groups','zone_f.gpkg'), layer='edges')
    nodes = gpd.read_file(path.join(gis_data_path, 'streets', 'sub_groups','zone_f.gpkg'), layer='nodes')
    
    # recreate edges in format for osmnx
    edges_f = graph_funcs.edge_gdf_format_from_gpkg(edges)
    nodes_f = graph_funcs.node_gdf_format_from_gpkg(nodes)

    # add layer allocation
    edges_f['layer_allocation'] = graph_funcs.allocate_group_height(nodes_f, edges_f, rotation_val=0)

    # calculate integral bearing difference
    edges_geometry = edges_f['geometry'].to_numpy()
    edges_f['integral_bearing_diff'] = graph_funcs.calculate_integral_bearing_difference(edges_geometry)

    # create graph and save edited
    G_f = ox.graph_from_gdfs(nodes_f, edges_f)

    # save as graphml
    ox.save_graphml(G_f, filepath=path.join(gis_data_path, 'streets', 'sub_groups','zone_f_gen.graphml'))
    
    # Save geopackage for import to QGIS and momepy
    ox.save_graph_geopackage(G_f, filepath=path.join(gis_data_path, 'streets', 'sub_groups','zone_f_gen.gpkg'), directed=True)

    print('--------FINISHED ZONE F---------')


if __name__ == '__main__':
    main()