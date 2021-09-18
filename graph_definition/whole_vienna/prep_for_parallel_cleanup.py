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
    G = ox.load_graphml(filepath=path.join(gis_data_path, 'streets', 'regrouping_3.graphml'))

    # edges to delete
    del_edges = [(30882785, 1382306305, 0), (30882787, 30882785, 0), (30882776, 30882787, 0),
                (93279698, 93279707, 0), (3703776564, 93279698, 0), (1362664821, 3703776564, 0),
                (341477461, 1362664821, 0), (30685741, 341477461), (77506774, 30685741, 0), (30685754, 77506774, 0),
                (30685739, 30685754, 0), (30685736, 30685739, 0), (30685749, 30685736, 0), (30685748, 30685749, 0), 
                (5919412735, 30685748, 0), (30685749, 30685743, 0), (27375757, 93279698, 0)]
    G.remove_edges_from(del_edges)

    # nodes to delete
    del_nodes = [30685749]
    G.remove_nodes_from(del_nodes)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    new_edge = (27375731, 30882785, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='1360'))

    # extend edges
    edges_to_extend = [(294369999, 30882776, 0), (83686174, 93279707, 0), (249199619, 93279698, 0),
                       (3703776564, 77506145, 0), (341477461, 249202894, 0), (17322862, 30685741, 0),
                       (77506771, 77506774, 0), (30685754, 30692823, 0), (47046953, 30685739, 0),
                       (47046949, 30685736, 0), (30685748, 47046938, 0)]
    nodes, edges = graph_funcs.extend_edges(nodes, edges, edges_to_extend)
    
    # graph 
    G = ox.graph_from_gdfs(nodes, edges)

    # saves as graphml
    ox.save_graphml(G, filepath=path.join(gis_data_path, 'streets', 'hand_edit_parallel.graphml'))

    # Save geopackage for import to QGIS
    ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'streets', 'hand_edit_parallel.gpkg'), directed=True)

if __name__ == '__main__':
    main()