from platform import node
import osmnx as ox
import geopandas as gpd
import networkx as nx
from shapely.geometry.point import Point
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

    '''--------------------------------------ZONE A----------------------------------'''
    # # TODO:
    ###### cleanup 1

    # edges to delete cleanup 1
    # del_edges = [(1243047818, 33236701, 0)]
    
    # G.remove_edges_from(del_edges)

    # nodes to delete
    del_nodes = [33236701]
    G.remove_nodes_from(del_nodes)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)


    # # extend edges
    # edges_to_extend = [(294369999, 30882776, 0), (83686174, 93279707, 0), (249199619, 93279698, 0),
    #                    (3703776564, 77506145, 0), (341477461, 249202894, 0), (17322862, 30685741, 0),
    #                    (77506771, 77506774, 0), (30685754, 30692823, 0), (47046953, 30685739, 0),
    #                    (47046949, 30685736, 0), (30685748, 47046938, 0)]
    # nodes, edges = graph_funcs.extend_edges(nodes, edges, edges_to_extend)

    G = ox.graph_from_gdfs(nodes, edges)


    '''--------------------------------------ZONE C----------------------------------'''
    # TODO: FIX NODE 27377268 and connectivity around 199633
    ###### cleanup 1

    # edges to delete cleanup 1
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

    G = ox.graph_from_gdfs(nodes, edges)


    ###### cleanup 2
    # edges to delete cleanup 2
    del_edges = [(27377268, 2526546837, 0), (2526546837, 144608149, 0), (144608149, 62592209, 0),
                        (62592209, 62598141, 0), (62598141, 60584137, 0), (60584137, 59640976, 0),
                        (59640976, 53169050, 0), (53169050, 60584155, 0), (60584155, 60584168, 0),
                        (60584168, 59640975, 0), (59640975, 78431390, 0), (78431390, 27377267, 0), 
                        (27377267, 86002346, 0), (86002346, 8790237562, 0)]
    
    G.remove_edges_from(del_edges)

    # nodes to delete
    del_nodes = [2526546837]
    G.remove_nodes_from(del_nodes)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    # extend edges
    edges_to_extend = [(144608149, 144608561, 0), (62598139, 62598141, 0), (60584137, 2309659095, 0),
                       (53169050, 60569923, 0), (213451282, 60584155, 0), (213451284, 60584168, 0),
                       (62598523, 59640975, 0), (78431381, 78431390, 0), (301933391, 27377267, 0),
                       (2309659180, 86002346, 0)]

    nodes, edges = graph_funcs.extend_edges(nodes, edges, edges_to_extend)

    G = ox.graph_from_gdfs(nodes, edges)
 
    ###### cleanup 3a
    # edges to delete
    del_edges = [(3249726549, 3249726527, 0), (192000050, 3080800578, 0), (306657000, 306658147, 0),
                (6739899006, 1521925366, 0), (1521925366, 306657000, 0), (86057499, 306657320, 0),
                (6739899006, 86057502, 0), (86057502, 6739898989, 0), (295414245, 61831870, 0),
                (295414234, 295414245, 0), (295414216, 295414234, 0), (199665, 6853942832, 0)]
    G.remove_edges_from(del_edges)

    # nodes to delete 
    del_nodes = [6739847856, 199632, 115450165, 306657000, 86057502]
    G.remove_nodes_from(del_nodes)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    # join existing nodes
    new_edge = (3249726549, 3249726527, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='1165'))

    # join existing nodes
    new_edge = (6739899006, 6739898989, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='1185'))

    # merge 1185 and 813

    # extend edges
    edges_to_extend = [(306658147, 306657060, 0), (295414245, 103656328, 0), (103656325, 295414234, 0)]
    nodes, edges = graph_funcs.extend_edges(nodes, edges, edges_to_extend)

    # new edge with osmid
    osmid = 306657320
    new_point = Point(16.394792537909574, 48.175204960544534)
    nodes, edges = graph_funcs.new_edge_osmid_to_point(nodes, edges, osmid, new_point, '1166')

    # new edge with osmid
    osmid = 1521925366
    new_point = Point(16.398295671447663, 48.17501003653269)
    nodes, edges = graph_funcs.new_edge_osmid_to_point(nodes, edges, osmid, new_point, '1195')

    # join existing nodes
    new_edge = (199665, 6853942832, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='1165'))

    G = ox.graph_from_gdfs(nodes, edges)

    # ###### cleanup 3b
    # # edges to delete
    # del_edges = [(86057499, 306657320, 0)]
    # G.remove_edges_from(del_edges)

    # # nodes to delete 
    # # del_nodes = [6739847856, 199632, 115450165, 306657000, 1521925366]
    # # G.remove_nodes_from(del_nodes)

    # # get_gdfs
    # nodes, edges = ox.graph_to_gdfs(G)


    # # extend edges
    # edges_to_extend = [(306658147, 306657060, 0)]

    # nodes, edges = graph_funcs.extend_edges(nodes, edges, edges_to_extend)

    # G = ox.graph_from_gdfs(nodes, edges)

    # saves as graphml
    ox.save_graphml(G, filepath=path.join(gis_data_path, 'streets', 'hand_edit_parallel.graphml'))

    # Save geopackage for import to QGIS
    ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'streets', 'hand_edit_parallel.gpkg'), directed=True)

if __name__ == '__main__':
    main()