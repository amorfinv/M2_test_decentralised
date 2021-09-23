import osmnx as ox
import geopandas as gpd
import networkx as nx
from shapely.geometry.point import Point
import graph_funcs
from os import path
from shapely import ops
import pandas as pd
from momepy import COINS
# use osmnx environment here

'''
'''

def main():

    # working path
    gis_data_path = 'gis'

    # Load MultiDigraph from cleanup_graph_2.py
    G = ox.load_graphml(filepath=path.join(gis_data_path, 'streets', 'hand_edit_parallel.graphml'))

    # remove some nodes
    del_nodes = [279621271, 277189392, 68913897, 294142171, 252370884, 105721823, 162757388, 2356191820, 
                162728583, 162731907, 162741976, 162684983, 162699034, 672177, 5697083, 392653, 298950959,
                33236699, 1494338462, 3004354233, 1972051, 1972036, 25280893, 25280872, 25280871, 25280870, 
                270510951, 270510952, 1212970742, 25280877, 8790237563, 8790237562, 1086176388, 706464962,
                29006424, 553281114, 553281132, 61837807, 76462549, 3080800569, 110324130, 110319068,
                1615863309, 3411602871, 17323951, 33469785, 101559759, 1195672779, 99111192, 60632372,
                366817080, 66835525, 93956250, 66835056, 251471819, 61920480, 101205989, 61920480, 101205989]
    
    G.remove_nodes_from(del_nodes)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    # join existing nodes
    new_edge = (1835762311, 1940108506, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='207'))

    # join existing nodes
    new_edge = (60957681, 392318, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='1016'))

    # new edge with osmid
    osmid = 61922604
    new_point = Point(16.31818972894718, 48.23890226382028)
    nodes, edges = graph_funcs.new_edge_osmid_to_point(nodes, edges, osmid, new_point, '249')

    # join existing nodes
    new_edge = (199557, 68916904, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='166'))   

    # new edge with osmid
    osmid = 289862055
    new_point = Point(16.310158481225308, 48.22836225850583)
    nodes, edges = graph_funcs.new_edge_osmid_to_point(nodes, edges, osmid, new_point, '166')

    # create graph
    G = ox.graph_from_gdfs(nodes, edges)

    ########### next cleanup
    del_nodes = [34978490, 68899674, 34978500, 5697118]

    G.remove_nodes_from(del_nodes)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    # join existing nodes
    new_edge = (199556, 213287619, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='166'))

    # join existing nodes
    new_edge = (6266012519, 378478, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='237'))   

    # join existing nodes
    new_edge = (378478, 9697671, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='237'))   

    # join existing nodes
    new_edge = (9697671,5866359, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='237')) 

    # join existing nodes
    new_edge = (600206222,3283142873, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='237')) 

    # join existing nodes
    new_edge = (34166934,64976837, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='237')) 

    # new edge with osmid
    osmid = 25280874
    new_point = Point(16.314412429461793, 48.1769369354651)
    nodes, edges = graph_funcs.new_edge_osmid_to_point(nodes, edges, osmid, new_point, '166')

    # create graph
    G = ox.graph_from_gdfs(nodes, edges)


    ########### next cleanup
    del_nodes = [5067529147]

    G.remove_nodes_from(del_nodes)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    # join existing nodes
    new_edge = (78206150, 252278546, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='924'))

    # join existing nodes
    new_edge = (252278546, 199579, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='924'))   

   # join existing nodes
    new_edge = (86002348, 61836724, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='856'))  

    # join existing nodes
    new_edge = (61836724, 123752235, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='856')) 

    # create graph
    G = ox.graph_from_gdfs(nodes, edges)

    ##### next cleanup for disemination

    # temperoary remove an edge for coins
    del_edges = [(27027315, 199656, 0)]
    G.remove_edges_from(del_edges)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)
    
    ## run a new coins
    coins_obj = COINS(edges)
    edges['stroke_group'] = coins_obj.stroke_attribute()

    # set intial group directions to get everything in order
    init_edge_directions = graph_funcs.get_first_group_edges(G, edges)
    edges = graph_funcs.set_direction(nodes, edges, init_edge_directions)

    # simplify graph and get a new coins
    nodes, edges = graph_funcs.simplify_graph(nodes, edges, angle_cut_off=120)
    
    ## run a new coins
    coins_obj = COINS(edges)
    edges['stroke_group'] = coins_obj.stroke_attribute()

    # readd edge delted above
    # create graph
    G = ox.graph_from_gdfs(nodes, edges)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)  

    # split group at an edge
    edges = graph_funcs.split_group_at_node(edges, 33469810, '29')

    G = ox.graph_from_gdfs(nodes, edges)

    # new edge with osmid
    osmid = 27027315
    new_point = Point(16.41993200607308,48.19151248104808)
    nodes, edges = graph_funcs.new_edge_osmid_to_point(nodes, edges, osmid, new_point, '1366')

    # create graph
    G = ox.graph_from_gdfs(nodes, edges)

    # saves as graphml
    ox.save_graphml(G, filepath=path.join(gis_data_path, 'streets', 'boundary_clean.graphml'))

    # Save geopackage for import to QGIS
    ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'streets', 'boundary_clean.gpkg'), directed=True)


if __name__ == '__main__':
    main()