import networkx as nx
from networkx.classes.function import non_edges
import pandas as pd
import numpy as np
import osmnx as ox
import geopandas as gpd
from os import path
from scipy.sparse import csr_matrix
from pympler.asizeof import asizeof as sizeof

"""
builds easy mappings from node_id to edge_id and vice versa

    edge_id_array: is a numpy array from 0 to m (m+1 = number of edges)
    edge_to_uv_array: is a numpy array of tuples (u,v). The index of the array (0 to m) is the edge_id

    node_id_array: is a numpy array from 0 to n (n+1 = number of nodes)
    uv_to_edge_matrix: is a sparse matrix with (u,v) as the index and the edge_id as the value
    edge_succesors_array: is a numpy array of where the index of the row is the node_id and the 
                            index of the column are the children of that node. The size of the array
                            is (n + 1) x 3 (3 is just for this graph). There should always be a child
                            for each node because there are no dead-ends. However if any of the values
                            in a row are 5000, then that means that you should stop searching for children.

"""

def main():
    # working path
    gis_data_path = 'gis'

    # Load MultiDigraph
    G = ox.load_graphml(filepath=path.join(gis_data_path, 'renamed_graph.graphml'))

    # convert to gdf
    nodes, edges = ox.graph_to_gdfs(G)

    # create the node id array
    node_id_array = np.arange(len(nodes), dtype='uint16')

    # get list of succesors that is length of node_id_array x 3 (3 is just for this graph)
    edge_succesors_array = np.empty((len(node_id_array), 3), dtype='uint16')

    for node in node_id_array:
        successors = list(G.successors(node))
        len_successors = len(successors)
        for idx in range(0,3):
            # 3 is just for this graph
            if idx < len_successors:
                edge_succesors_array[node, idx] = successors[idx]
            else:
                # choose 5000 because there are less than 5000 nodes in the graph
                edge_succesors_array[node, idx] = 5000

    # get the edge_id to uv mapping
    edge_to_uv_array = np.array([(index[0], index[1])for index in edges.index.values], dtype='uint16,uint16')

    # build the edge_id_array
    edge_id_array = np.arange(len(edge_to_uv_array), dtype='uint16')

    # intialize an empty matrix that is the len of nodes
    adj_matrix = np.zeros((len(nodes), len(nodes)), dtype='uint16')

    # populate this matrix (row=node_id, col=node_id) and value at row, col is edge_id
    for j, ids in enumerate(edge_to_uv_array):
        adj_matrix[ids[0], ids[1]] = j
    
    # convert to sparse matrix to get the (u,v) to edge_id mapping
    uv_to_edge_matrix = csr_matrix(adj_matrix, dtype=np.uint16)

    # get size of the objects
    print('size of edge_id_array:', sizeof(edge_id_array))
    print('size of edge_to_uv_array:', sizeof(edge_to_uv_array))

    print('size of node_id_array:', sizeof(node_id_array))
    print('size of uv_to_edge_matrix:', sizeof(uv_to_edge_matrix))

    print('size of edge_succesors_array:', sizeof(edge_succesors_array))

    # # sum them all
    # print('total size of all objects:', sizeof(edge_id_array) + sizeof(edge_to_uv_array) + sizeof(node_id_array) + sizeof(uv_to_edge_matrix))

if __name__ == '__main__':
    main()