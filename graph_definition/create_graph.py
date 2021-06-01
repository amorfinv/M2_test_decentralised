import osmnx as ox
import networkx as nx
from matplotlib import pyplot as plt
from funcs import coins, graph_funcs
from os import path

# use osmnx environment here

def main():

    # working paths
    gis_data_path = path.join('gis','data')

    # convert shapefile to shapely polygon
    center_poly = graph_funcs.poly_shapefile_to_shapely(path.join(gis_data_path, 'street_info', 'poly_shapefile.shp'))

    # create MultiDigraph from polygon
    G = ox.graph_from_polygon(center_poly, network_type='drive', simplify=True)

    # save as osmnx graph
    ox.save_graphml(G, filepath=path.join(gis_data_path, 'street_graph', 'raw_streets.graphml'))
    ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'street_graph', 'raw_graph.gpkg'))

    # remove unconnected streets and add edge bearing attrbute 
    G = graph_funcs.remove_unconnected_streets(G)
    ox.add_edge_bearings(G)

    # manually remove some nodes and edges
    nodes_to_remove = [33144416, 33144419, 33182073, 33344823, 271016303, 393097, 33345333, 33345331, 83345330, 3963787755, 33345337, 26405238,
                       33345330, 251207325, 33344804]
    G.remove_nodes_from(nodes_to_remove)
    G.remove_edge(33345319, 29048469)

    # convert graph to geodataframe
    g = ox.graph_to_gdfs(G)

    # # get node and edge geodataframe
    nodes = g[0]
    edges = g[1]

    # remove double two way edges
    edges = graph_funcs.remove_two_way_edges(edges)

    # remove non parallel opposite edges (or long way)
    edges = graph_funcs.remove_long_way_edges(edges)

    # allocated edge height based on cardinal method
    layer_allocation, _ = graph_funcs.allocate_edge_height(edges, 0)

    # Assign layer allocation to geodataframe
    edges['layer_height'] = layer_allocation

    # # add interior angles at all intersections
    edges = graph_funcs.add_edge_interior_angles(edges)

    # Perform COINS algorithm to add stroke groups
    coins.COINS(edges)

    # set directionality of edges
    graph_funcs.set_direction(edges)

    # create graph and save edited
    G = ox.graph_from_gdfs(nodes, edges)
    # save as osmnx graph
    ox.save_graphml(G, filepath=path.join(gis_data_path, 'street_graph', 'processed_graph.graphml'))

    # Save geopackage for import to QGIS and momepy
    ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'street_graph', 'processed_graph.gpkg'))

    # save projected
    G_projected = ox.project_graph(G)
    ox.save_graph_geopackage(G_projected, filepath=path.join(gis_data_path, 'street_graph', 'projected_graph.gpkg'))

    # save csv for reference
    edges.to_csv(path.join(gis_data_path, 'street_graph', 'edges.csv'))
    nodes.to_csv(path.join(gis_data_path, 'street_graph', 'nodes.csv'))

if __name__ == '__main__':
    main()