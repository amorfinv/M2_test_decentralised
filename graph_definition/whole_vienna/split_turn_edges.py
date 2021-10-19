import osmnx as ox
import geopandas as gpd
from os import path
import numpy as np
from time import time
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import split
import graph_funcs
# use osmnx environment here

'''
Prepare graph for height allocation genetic algorithm.
'''

def main():

    # working path
    gis_data_path = 'gis'

    # Load MultiDigraph from create_graph.py
    G = ox.load_graphml(filepath=path.join(gis_data_path, 'streets', 'prep_height_allocation.graphml'))

    # convert to gdf
    nodes, edges = ox.graph_to_gdfs(G)

    # get list of edges to split from QGIS
    turning_edges = gpd.read_file(path.join(gis_data_path, 'streets', 'turning_groups.gpkg'))
    turning_edges = graph_funcs.edge_gdf_format_from_gpkg(turning_edges)
    edge_set = set(turning_edges.index.values)

    # Turning excludes
    turning_excludes = {# (291221138, 291221319, 0),  # try to split
                        (247748343, 48097254, 0),
                        (48097197, 34765361, 0),
                        (34591546, 2396698510, 0),
                        (294369773, 294369999, 0),
                        (33183721, 2296833087, 0),
                        (78120039, 1168799412, 0),
                        (33208032, 33208029, 0),
                        (199659, 1445735725, 0),
                        (33208044, 33208082, 0),  # failed to work with split_edges_at_curve
                        (5697076, 1986051, 0),     # not the greatest with split_edges_at_curve move to split_edges_at largest angle
                        (66863208, 60019309, 0),     # not the greatest with split_edges_at_curve move to split_edges_at largest angle
                        (46959602, 30675716, 0),    # not the greatest with split_edges_at_curve move to split_edges_at largest angle
                        (31259951, 268703540, 0),  # not the greatest with split_edges_at_curve move to split_edges_at largest angle
                        (33182067, 378700, 0),  # not the greatest with split_edges_at_curve move to split_edges_at largest angle
                        (2455074401, 685161, 0), # not the greatest with split_edges_at_curve move to split_edges_at largest angle
                        (199745, 199747, 0),    # not the greatest with split_edges_at_curve move to split_edges_at largest angle
                        (199747, 199753, 0),    # not the greatest with split_edges_at_curve move to split_edges_at largest angle
                        (3665855676, 306664777, 0)  # not the greatest with split_edges_at_curve or split_edges_at largest angle do with split_edges_at_idx
                        }
    # remove turning excludes from edge_list
    edge_set = edge_set.symmetric_difference(turning_excludes)

    # turning include{}
    turning_includes = {#(1200092857, 274590482, 0), removed as it was in the border of the constrained airspae
                        (59766917, 123673053, 0),
                        (33344348, 277838181, 0),
                        (60211429, 60211434, 0)}

    # add turning includes edges
    edge_set = edge_set.union(turning_includes)

    # split edges at curve
    edge_list = list(edge_set)
    nodes, edges = graph_funcs.split_edges_at_curve(nodes, edges, edge_list, new_group=True)

    # split edges at largest angle
    edge_list = [(33208044, 33208082, 0), (5697076, 1986051, 0), (66863208, 60019309, 0),
                 (46959602, 30675716, 0), (31259951, 268703540, 0), (33182067, 378700, 0),
                 (2455074401, 685161, 0), (199745, 199747, 0), (199747, 199753, 0)]
    nodes, edges = graph_funcs.split_edges_at_largest_angle(nodes, edges, edge_list, new_group=True)

    # split with index
    edge_dict = {(3665855676, 306664777, 0):20, (33183721, 2296833087, 0): 2, (33182887, 17322928, 0): 11}
    nodes, edges = graph_funcs.split_edges_at_idx(nodes, edges, edge_dict, new_group=True)

    # split edges at centroid
    edge_list = [(33208032, 33208029, 0), (28150494, 311045196, 0), (92739182, 92736477, 0), 
                 (123673053, 2061414510, 0), (2383639011, 8074083217, 0), (393334, 392479, 0),
                 (394906, 199572, 0), (60637360, 685051, 0), (103659103, 27027755, 0)]
    nodes, edges = graph_funcs.split_edges_at_centroid(nodes, edges, edge_list, new_group=True)

    # create graph
    G = ox.graph_from_gdfs(nodes, edges)

    # save graphml
    ox.save_graphml(G, filepath=path.join(gis_data_path, 'streets', 'prep_height_allocation_new.graphml'))

    # save gpkg
    ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'streets', 'prep_height_allocation_new.gpkg'), directed=True)

if __name__ == '__main__':
    main()