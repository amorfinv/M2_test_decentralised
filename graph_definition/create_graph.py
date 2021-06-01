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

    # set directionality of groups with one edge
    edge_directions = [(1629378075, 378728, 0),     # Edge 0
                    (264055540, 33144941, 0),   # Edge 1
                    (33345285, 33345298, 0),    # Edge 2
                    (60631071, 401838, 0),      # Edge 3
                    (33144366, 33143888, 0),    # Edge 4
                    (394751, 392442, 0),        # Edge 5
                    (33144601, 33144648, 0),    # Edge 6
                    (655012, 33144500, 0),      # Edge 7
                    (3312560802, 33144366, 0),  # Edge 8
                    (264055538, 264055539, 0),  # Edge 9
                    (29048460, 33345320, 0),    # Edge 10
                    (33143898, 33143888, 0),    # Edge 11
                    (64971266, 64970487, 0),    # Edge 12
                    (251523470, 33345297, 0),   # Edge 13
                    (29048469, 64972028, 0),    # Edge 14
                    (283324403, 358517297, 0),  # Edge 15
                    (33345286, 33345303, 0),    # Edge 16
                    (33344817, 401838, 0),      # Edge 17
                    (33344807, 33144550, 0),    # Edge 18
                    (33144633, 33144941, 0),    # Edge 19
                    (33144471, 33144422, 0),    # Edge 20
                    (33344808, 33144550, 0),    # Edge 21
                    (33144566, 33144555, 0),    # Edge 22
                    (33144560, 283324407, 0),   # Edge 23
                    (33144659, 33144655, 0),    # Edge 24
                    (33144605, 33144652, 0),    # Edge 25
                    (33144616, 33144655, 0),    # Edge 26
                    (33144755, 33144759, 0),    # Edge 27
                    (33344802, 33344805, 0),    # Edge 28
                    (33344812, 33344811, 0),    # Edge 29
                    (33344809, 2423479559, 0),  # Edge 30
                    (33345299, 33344825, 0),    # Edge 31
                    (33344824, 33344825, 0),    # Edge 32
                    (33345310, 33345289, 0),    # Edge 33
                    (64975949, 33345310, 0),    # Edge 34
                    (64975551, 33345319, 0),    # Edge 35
                    (245498401, 245498398, 0),  # Edge 36
                    (33144637, 320192043, 0)    # Edge 37                   
                    ]
    # reoroder edge geodatframe
    edges = graph_funcs.set_direction(edges, edge_directions)

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