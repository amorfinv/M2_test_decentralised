import osmnx as ox
import networkx as nx
from matplotlib import pyplot as plt
from funcs import coins_fancy, graph_funcs
from os import path
from hierholzer import hierholzer
from directionality import calcDirectionality
import copy

# use osmnx environment here

def main():

    # working paths
    gis_data_path = path.join('gis','data')
    
    # convert shapefile to shapely polygon
    center_poly = graph_funcs.poly_shapefile_to_shapely(path.join(gis_data_path, 'street_info', 'new_poly_shapefile.shp'))
    
    # create MultiDigraph from polygon
    G = ox.graph_from_polygon(center_poly, network_type='drive', simplify=True)
    
    # save as osmnx graph
    ox.save_graphml(G, filepath=path.join(gis_data_path, 'street_graph', 'raw_streets.graphml'))
    ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'street_graph', 'raw_graph.gpkg'))
    
    # remove unconnected streets and add edge bearing attrbute 
    G = graph_funcs.remove_unconnected_streets(G)
    ox.add_edge_bearings(G)
    
    #ox.plot.plot_graph(G)
    
    # manually remove some nodes and edges to clean up graph
    nodes_to_remove = [33144419, 33182073, 33344823, 83345330, 33345337,
                       33345330, 33344804, 30696018, 5316966255, 5316966252, 33344821,
                       2335589819, 245498397, 287914700, 271016303, 2451285012, 393097]
    edges_to_remove = [(291088171, 3155094143), (60957703, 287914700), (2451285012, 287914700),
                       (25280685, 30696019), (30696019, 25280685), (392251, 25280685), 
                       (25280685, 392251), (33301346, 1119870220),  
                       (33345331, 33345333), (378699, 378696), (378696, 33143911), 
                       (33143911, 33144821), (264061926, 264055537), (33144706, 33144712),
                       (33144712, 33174086), (33174086, 33144719), (33144719, 92739749),
                       (33345319, 29048469), (287914700, 60957703), (213287623, 251207325),
                       (251207325, 213287623)]
    G.remove_nodes_from(nodes_to_remove)
    G.remove_edges_from(edges_to_remove)
    
    #ox.plot.plot_graph(G)
    
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
    coins_obj = coins_fancy.COINS(edges)
    edges['stroke_group'] = coins_obj.stroke_attribute()
    group_gdf = coins_obj.stroke_gdf()
    
    init_edge_directions = graph_funcs.get_first_group_edges(G, group_gdf, edges)
    
    # Apply direction algorithm
    edge_directions = calcDirectionality(group_gdf, nodes, init_edge_directions)
    
    # reoroder edge geodatframe
    edges = graph_funcs.set_direction(edges, edge_directions)
    
    # create graph and save edited
    G = ox.graph_from_gdfs(nodes, edges)
    
    # get undirected graph
    G_un = ox.get_undirected(G)
    
    # get eulerized graph
    G_euler = nx.eulerize(G_un)
    
    # add geometry info to added edges
    ox.distance.add_edge_lengths(G_euler)
    
    
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