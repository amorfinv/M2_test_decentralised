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
import itertools as it

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
    edges = flow_control_split(edges)

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
    Code takes stroke groups larger than threshold and splits them to be the average length
    """
    # make split threshold 1.5 times the average length
    split_threshold = average_length * 1.5

    # make a copy
    edges_gdf = edges.copy()
    
    stroke_length = {}
    flow_group_dict = {}
    flow_group_counter = 1

    # read json file with stroke group lengths
    with open(path.join('stroke_length.json'), 'r') as f:
        stroke_length = json.load(f)

    # for loop to go through
    flow_group_dict[0] = []
    jj = 0
    ii = 0
    for stroke in stroke_length.keys():
        # print('Looking at stroke: ', stroke)
        # create a datafrme with only the stroke group
        stroke_gdf = edges_gdf.loc[edges_gdf['stroke_group'] == stroke]

        # get the length of the stroke
        length_stroke = stroke_gdf['length'].sum()
 
        # get the stroke indices
        stroke_edge_indices, length_edges_list = order_stroke_group_edges(stroke_gdf)

        # if the length is less than the ignore threshold, then ignore
        if length_stroke < ignore_threshold:
            # print(f'stroke group {stroke} is less than {ignore_threshold} and will be added to split group 0')

            # these will get a flow group of 0
            for edge_index in stroke_edge_indices:

                # get the actual edge index
                index = (edge_index[0], edge_index[1], 0)
                
                # add to the dictionary
                flow_group_dict[0].append(index)
            
            # restart the for loop
            continue

        # check if length is less than the split threshold
        # This means that you just create a new flow group
        if length_stroke < split_threshold:
            # print(f'stroke group {stroke} is less than {split_threshold} and will be added to split group {flow_group_counter}')
            # get the new stroke group
            new_stroke = flow_group_counter

            # initliaze the flow group as an empty dictionary
            flow_group_dict[new_stroke] = []

            # this group will get a new flow group with all its edges
            for edge_index in stroke_edge_indices:

                # get the actual edge index
                index = (edge_index[0], edge_index[1], 0)
                
                # add to the dictionary
                flow_group_dict[new_stroke].append(index)
            
            # extend flow group counter
            flow_group_counter += 1

        else:

            # this happens when the stroke length is greater than the split threshold so
            # you need to split the stroke into multiple flow groups            
            num_splits = round(length_stroke / average_length)

            if num_splits > 8:
                print(f'Stroke group {stroke} has {num_splits} splits')
                print('------------------------------------------------------------------------------------')
                jj += 1
                continue

            if len(stroke_edge_indices) <= num_splits:
                print(f'Stroke group {stroke} has {len(stroke_edge_indices)} edges and {num_splits} splits')
                print('------------------------------------------------------------------------------------')
                ii += 1
                continue

            # get approximate length of each split
            approximate_length = length_stroke / num_splits
            
            # print(f'stroke group {stroke} is greather than {split_threshold} and will be attempted to be split into {num_splits} flow groups')

            # find the place to
            # print(stroke_edge_indices)
            # print(length_edges_list)
            # find the place to split the stroke so that the sum of the lengths is near approximate length
            split_indices = find_split_indices(stroke_edge_indices, length_edges_list, approximate_length, num_splits)
            # print(split_indices)
            split_groups_new = []

            # add the first split group to the dictionary
            first_split_group_edges = stroke_edge_indices[0:split_indices[0]]
            split_groups_new.append(first_split_group_edges)

            for index in range(0, len(split_indices)-1):
                edges_in_group = stroke_edge_indices[split_indices[index]:split_indices[index+1]]
                split_groups_new.append(edges_in_group)

            # add the last split group to the dictionary
            last_split_group_edges = stroke_edge_indices[split_indices[-1]:]
            split_groups_new.append(last_split_group_edges)

            # print(split_groups_new)
            # list1 = stroke_edge_indices
            # indices = [(0, 2), (3, 5), (6,8)]
            # print([list1[s:e+1] for s,e in indices])

            # loop through the split groups
            for split_group in split_groups_new:
                
                # get the new stroke group
                new_stroke = flow_group_counter

                # initliaze the flow group as an empty dictionary
                flow_group_dict[new_stroke] = []

                # loop through the edge
                for edge_index in split_group:

                    # get the actual edge index
                    index = (edge_index[0], edge_index[1], 0)
                    
                    # add to the dictionary
                    flow_group_dict[flow_group_counter].append(index)

                # extend flow group counter
                flow_group_counter += 1
            

        # print('------------------------------------------------------------------------------------')

    print(f'There are {jj} strokes with large splits')
    print(f'There are {ii} strokes with too small splits')
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

    return edges_gdf

def find_split_indices(stroke_edge_indices, length_edges_list, approximate_length, num_splits):
    """
    Find the split indices. Brute force approach. It uses the number of splits and them splits
    everywhere it can. Then it decides which split to use based on the error of approximate length

    Parameters
    ----------
    stroke_edge_indices : list
        list of edges in the stroke group
    length_edges_list : list
        list of lengths of the edges in the stroke group
    approximate_length : int
        approximate length of the stroke group
    num_splits : int
        number of splits to make

    Returns
    -------
    split_indices : list
        list of indices where the stroke should be split
    """
    # get the number of edges that need to be split
    num_edges = len(stroke_edge_indices)

    # get the number possible split locations
    possible_split_locations = list(range(1, num_edges))

    # get the possible split indices
    poss_split_indices = list(it.combinations(possible_split_locations, num_splits-1))

    # loop through the possible split indices
    split_choices_error = []
    split_choices_length = []
    for split_indices in poss_split_indices:
        # loop through split_indices
        split_index = list(split_indices)

        # sum the length of edges in length_edges_list using split index
        first_index = split_index[0]
        last_index = split_index[-1]

        split_choice_error = []
        split_choice_length = []

        # sum the first index
        first_index_sum = sum(length_edges_list[:first_index])
        first_index_error = abs(first_index_sum - approximate_length)

        split_choice_length.append(first_index_sum)
        split_choice_error.append(first_index_error)

        # sum the in between indices
        for index in range(0, len(split_index)-1):
            bw_index_sum = sum(length_edges_list[split_index[index]:split_index[index+1]])
            bw_index_error = abs(bw_index_sum - approximate_length)

            split_choice_length.append(bw_index_sum)
            split_choice_error.append(bw_index_error)

        # sum the last index
        last_index_sum = sum(length_edges_list[last_index:])
        last_index_error = abs(last_index_sum - approximate_length)

        split_choice_length.append(last_index_sum)
        split_choice_error.append(last_index_error)

        # append list to split choices
        split_choices_error.append(sum(split_choice_error))
        split_choices_length.append(split_choice_length)

    # get the minimum split choice
    min_split_choice = min(split_choices_error)
    min_index = split_choices_error.index(min_split_choice)

    # print(split_choices_error[min_index])
    # print(split_choices_length[min_index])
    # print(poss_split_indices[min_index])

    # get the split choice
    best_split_choice_index = poss_split_indices[min_index] 

    return list(best_split_choice_index)


def order_stroke_group_edges(stroke_gdf):
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

    # get the length of the edges in the stroke as a dictionary
    length_edges_in_stroke = stroke_gdf['length'].to_dict()

    # loop through stroke_indices_ordered
    lengths_edges_stroke = [length_edges_in_stroke[(stroke_index[0], stroke_index[1], 0)] for stroke_index in stroke_indices_ordered]

    return stroke_indices_ordered, lengths_edges_stroke
        
if __name__ == '__main__':
    main()
