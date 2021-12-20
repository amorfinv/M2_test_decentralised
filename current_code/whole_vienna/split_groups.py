from unicodedata import name
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString, Polygon, LinearRing
import graph_funcs
from os import path
import numpy as np
import pandas as pd
from pympler import asizeof
import json

"""
Code to create groups for flow control

THIS IS THE FINAL GRAPH (maybe)
"""

def main():
    # working path
    gis_data_path = 'gis'

    # Load MultiDigraph from create_graph.py
    G = ox.load_graphml(filepath=path.join(gis_data_path, 'renamed_graph.graphml'))

    # convert to gdgs
    nodes, edges = ox.graph_to_gdfs(G)

    # rename node indices
    edges = flow_control_split2(edges)
    print(edges)
    # convert back to graph and save
    G = ox.graph_from_gdfs(nodes, edges)

    ox.save_graphml(G, filepath=path.join(gis_data_path, 'finalized_graph.graphml'))

    ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'finalized_graph.gpkg'), directed=True)

def flow_control_split2(edges, average_length=300, ignore_threshold=200):
    """
    Code takes stroke groups larger than threshold meters and splits them so they around
    """
    # make split threshold 1.5 times the average length
    split_threshold = average_length * 1.5

    # make a copy
    edges_gdf = edges.copy()
    
    stroke_length = {}

    # read json file with stroke group lengths
    with open(path.join('stroke_length.json'), 'r') as f:
        stroke_length = json.load(f)
    flow_group_dict = {}
    flow_group_counter = 1

    # for loop to go through
    flow_group_dict[0] = []

    for stroke in stroke_length.keys():
        length_stroke = stroke_length[stroke]

        # create a dataframe with only the stroke group
        stroke_gdf = edges_gdf.loc[edges_gdf['stroke_group'] == stroke]

        ###########################################
        # put indices in correct order
        stroke_indices = stroke_gdf.index.to_list()

        # remove the zero from the tuple
        stroke_indices = [(stroke_index[0], stroke_index[1]) for stroke_index in stroke_indices]
        stroke_indices_list = np.array([item for t in stroke_indices for item in t])

        # find which nodes appear once
        v,c = np.unique(stroke_indices_list, return_counts=True)
        unique_indices = v[c == 1]

        # find the first node of the stroke_indices
        for idx, stroke_index in enumerate(stroke_indices):
            if stroke_index[0] in unique_indices:
                first_node = stroke_index[0]
                index_first_edge = idx
                break
        
        node_to_find = stroke_indices[index_first_edge][1]
        stroke_indices_ordered = [stroke_indices[index_first_edge]]
        stroke_indices_check = stroke_indices.copy()
        # remove the first edge
        stroke_indices.remove(stroke_indices[index_first_edge])

        while stroke_indices:
            # find the node in the stroke_indices
            for idx, stroke_index in enumerate(stroke_indices_check):
                if stroke_index[0] == node_to_find:
                    node_to_find = stroke_index[1]
                    stroke_indices_ordered.append(stroke_index)
                    break
            # remove the node from the stroke_indices
            stroke_indices.remove(stroke_indices_check[idx])
        
        ###########################################

        # if the length is less than the ignore threshold, then ignore
        if length_stroke < ignore_threshold:

            # these will get a flow group of 0
            for index, edge in stroke_gdf.iterrows():
                flow_group_dict[0].append(index)

            continue
    
        
        # if the length is greater than the split threshold, then split the group
        if length_stroke > split_threshold:
            # check how many splits to do
            num_splits = round(length_stroke / average_length)

            # set length counter to 0
            length_counter = 0

            # get a counter for split
            split_counter = 0

            # loop through the stroke edges
            edge_list = []
            for edge_index in stroke_indices_ordered:
                # get the actual index
                index = (edge_index[0], edge_index[1], 0)

                edge_length = edges_gdf.loc[index, 'length']
                
                # if the length counter is greater than the average length, then split
                if length_counter + edge_length > average_length:

                    if length_stroke - (length_counter + edge_length) < 100:
                        a = 1
                    elif length_counter + edge_length < split_threshold:

                        edge_list.append(index)

                        # get the new stroke group
                        new_stroke = flow_group_counter

                        # add the new stroke to the dictionary
                        flow_group_dict[new_stroke] = edge_list

                        length_stroke = length_stroke - (length_counter + edge_length)

                        # extend flow group counter
                        flow_group_counter += 1

                        # reset the length counter
                        length_counter = 0

                        # initialize edge_list
                        edge_list = []

                    else:
                        # get the new stroke group
                        new_stroke = flow_group_counter

                        # add the new stroke to the dictionary
                        flow_group_dict[new_stroke] = edge_list

                        length_stroke = length_stroke - (length_counter + edge_length)
                
                        # extend flow group counter
                        flow_group_counter += 1

                        # reset the length counter
                        length_counter = 0

                        # initialize edge_list
                        edge_list = []
                
                
                # get the edge length
                length_counter += edge_length

                edge_list.append(index)
            
            # get the new stroke group
            new_stroke = flow_group_counter

            # add the new stroke to the dictionary
            flow_group_dict[new_stroke] = edge_list
    
            # extend flow group counter
            flow_group_counter += 1
        else:
            edge_list = []
            for index, edge in stroke_gdf.iterrows():
                edge_list.append(index)

            # get the new stroke group
            new_stroke = flow_group_counter

            # add the new stroke to the dictionary
            flow_group_dict[new_stroke] = edge_list

            # extend flow group counter
            flow_group_counter += 1       

    # edges_gdf['flow_group'] = np.zeros(len(edges_gdf), dtype=np.int64)
    edges_gdf['flow_group'] = np.ones(len(edges_gdf), dtype=np.int64)*(-1)

    # loop through dictionary
    for key in flow_group_dict.keys():
        # get the list of edges
        edge_list = flow_group_dict[key]

        # loop through the edges in the list
        for index in edge_list:
            # add the flow group to the dataframe
            edges_gdf.loc[index, 'flow_group'] = int(key)

    # # get minium value of flow group
    # min_flow_group = np.min(edges_gdf['flow_group'].to_numpy())
    # print(min_flow_group)
    return edges_gdf

def flow_control_split(edges, average_length=300, ignore_threshold=200):
    """
    Code takes stroke groups larger than threshold meters and splits them so they around
    """
    # make split threshold 1.5 times the average length
    split_threshold = average_length * 1.5

    # make a copy
    edges_gdf = edges.copy()

    # also remove some columns from dataframe
    edge_stroke = pd.DataFrame(edges_gdf['stroke_group'].copy())

    # get the length of stroke groups
    stroke_array = np.sort(np.unique(edge_stroke.loc[:, 'stroke_group'].to_numpy()).astype(np.int64))
    
    stroke_length = {}
    flow_group_dict = {}
    flow_group_counter = 1

    # read json file with stroke group lengths
    with open(path.join('stroke_length.json'), 'r') as f:
        stroke_length = json.load(f)

    # for loop to go through
    flow_group_dict[0] = []

    for stroke in stroke_length.keys():
        length_stroke = stroke_length[stroke]

        # create a datafrme with only the stroke group
        stroke_gdf = edge_stroke.loc[edge_stroke['stroke_group'] == stroke]

        # if the length is less than the ignore threshold, then ignore
        if length_stroke < ignore_threshold:

            # these will get a flow group of 0
            for index, edge in stroke_gdf.iterrows():
                flow_group_dict[0].append(index)

            continue
    
        
        # if the length is greater than the split threshold, then split thr group
        if length_stroke > split_threshold:
            # check how many splits to do
            num_splits = round(length_stroke / average_length)

            # set length counter to 0
            length_counter = 0

            # get a counter for split
            split_counter = 0

            # loop through the stroke edges
            edge_list = []
            for index, edge in stroke_gdf.iterrows():

                edge_length = edges_gdf.loc[index, 'length']
                
                # if the length counter is greater than the average length, then split
                if length_counter + edge_length > average_length:

                    if length_stroke - (length_counter + edge_length) < 100:
                        a = 1
                    elif length_counter + edge_length < split_threshold:

                        edge_list.append(index)

                        # get the new stroke group
                        new_stroke = flow_group_counter

                        # add the new stroke to the dictionary
                        flow_group_dict[new_stroke] = edge_list
                
                        # extend flow group counter
                        flow_group_counter += 1

                        # reset the length counter
                        length_counter = 0

                        # initialize edge_list
                        edge_list = []

                    else:
                        # get the new stroke group
                        new_stroke = flow_group_counter

                        # add the new stroke to the dictionary
                        flow_group_dict[new_stroke] = edge_list
                
                        # extend flow group counter
                        flow_group_counter += 1

                        # reset the length counter
                        length_counter = 0

                        # initialize edge_list
                        edge_list = []
                
                
                # get the edge length
                length_counter += edge_length

                edge_list.append(index)
            
            # get the new stroke group
            new_stroke = flow_group_counter

            # add the new stroke to the dictionary
            flow_group_dict[new_stroke] = edge_list
    
            # extend flow group counter
            flow_group_counter += 1
        else:
            edge_list = []
            for index, edge in stroke_gdf.iterrows():
                edge_list.append(index)

            # get the new stroke group
            new_stroke = flow_group_counter

            # add the new stroke to the dictionary
            flow_group_dict[new_stroke] = edge_list

            # extend flow group counter
            flow_group_counter += 1       

    # edges_gdf['flow_group'] = np.zeros(len(edges_gdf), dtype=np.int64)
    edges_gdf['flow_group'] = np.ones(len(edges_gdf), dtype=np.int64)*(-1)

    # loop through dictionary
    for key in flow_group_dict.keys():
        # get the list of edges
        edge_list = flow_group_dict[key]

        # loop through the edges in the list
        for index in edge_list:
            # add the flow group to the dataframe
            edges_gdf.loc[index, 'flow_group'] = int(key)

    # # get minium value of flow group
    # min_flow_group = np.min(edges_gdf['flow_group'].to_numpy())
    # print(min_flow_group)
    return edges_gdf

if __name__ == '__main__':
    main()
