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

def flow_control_split(edges, average_length=300, ignore_threshold=200, merge_threshold=190):
    """
    Code takes stroke groups larger than threshold and splits them to be the average length

    At the end code will check if any split groups are below the merge_threshold. It will then merge it with the adjacent 
    split group of the merge.
    """
    # make split threshold 1.5 times the average length
    split_threshold = average_length * 1.667

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
    for stroke in stroke_length.keys():
        # print('Looking at stroke: ', stroke)
        # create a datafrme with only the stroke group
        stroke_gdf = edges_gdf.loc[edges_gdf['stroke_group'] == stroke]

        # get the length of the stroke
        length_stroke = stroke_gdf['length'].sum()

        # get the stroke indices
        stroke_edge_indices, length_edges_list = order_group_edges(stroke_gdf)

        # get potential number of splits
        num_splits = round(length_stroke / average_length)

        ######## CHECK 1 ########
        # if the length is less than the ignore threshold, then ignore
        if length_stroke < ignore_threshold:
            if stroke == '80':
                stroke_edge_indices.append((1509, 1503))

            # print(f'stroke group {stroke} is less than {ignore_threshold} and will be added to split group 0')

            # these will get a flow group of 0
            for edge_index in stroke_edge_indices:

                # get the actual edge index
                index = (edge_index[0], edge_index[1], 0)
                
                # add to the dictionary
                flow_group_dict[0].append(index)
            
            # restart the for loop
            continue

        ######## CHECK 2 ########
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
        
        ######## CHECK 3 ########
        # Check if there are less than or equal to two edges in the stroke group. If yes then just add to its own flow group
        elif len(stroke_edge_indices) <= 2:
            #print(f'stroke group {stroke} has less than two edges and will be added to split group {flow_group_counter}')
            # get the new stroke group
            new_stroke = flow_group_counter

            # initliaze the flow group as an empty dictionary
            flow_group_dict[new_stroke] = []

            # this stroke group will get a new flow group with all its edges
            for edge_index in stroke_edge_indices:

                # get the actual edge index
                index = (edge_index[0], edge_index[1], 0)
                
                # add to the dictionary
                flow_group_dict[new_stroke].append(index)
            
            # extend flow group counter
            flow_group_counter += 1

        ######## CHECK 4 ########
        # check when the number of edges is 3 and the number of splits is less than or 2. This implies that edge lengths
        # would be quite small. So no need to split in this case
        elif len(stroke_edge_indices) == 3 and num_splits <= 2:
            # when a stroke has 3 edges and the number of splits is less than . This means that it is not worth splitting
            # print(f'stroke group {stroke} has three small edges and will be added to split group {flow_group_counter}')

            # get the new stroke group
            new_stroke = flow_group_counter

            # initliaze the flow group as an empty dictionary
            flow_group_dict[new_stroke] = []

            # this stroke group will get a new flow group with all its edges
            for edge_index in stroke_edge_indices:

                # get the actual edge index
                index = (edge_index[0], edge_index[1], 0)
                
                # add to the dictionary
                flow_group_dict[new_stroke].append(index)
            
            # extend flow group counter
            flow_group_counter += 1
        
        ######## CHECK 5 ########
        else:
            # this happens when the stroke length is greater than the split threshold and other things are not met

            ######## CHECK 5-a #######
            # Now do a general check to see if the number of edges is less than or equal to the number of splits
            # in this case adjust the number of splits to len(edges) - 2
            if len(stroke_edge_indices) <= num_splits:
                # adjust the number of splits to len(edges) - 2
                num_splits = len(stroke_edge_indices) - 2

            ######## CHECK 5-b #######
            if num_splits <= 8:  # if the number of splits is less than 8 then do the split with combinations

                # get approximate length of each split
                approximate_length = length_stroke / num_splits
                
                # find the place to split the stroke so that the sum of the lengths is near approximate length
                split_indices = find_split_indices(stroke_edge_indices, length_edges_list, approximate_length, num_splits, average_length)

                # get split groups new list depending on the split indices
                if len(stroke_edge_indices) > 3:
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
                else:
                    # when there is just one split which means len(edges) is 3
                    front_split_edges = stroke_edge_indices[:split_indices]
                    back_split_edges = stroke_edge_indices[split_indices:]

                    split_groups_new = [front_split_edges, back_split_edges]


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
            
            ######## CHECK 5-c #######
            # go in here when the number of splits is greater than 8 an combination is not possible because of time
            else:
                # print(f'Stroke group {stroke} has {len(stroke_edge_indices)} edges and {num_splits} splits')

                # now split manually. count the length of the edges in the stroke group and when it is larger
                # than the average length create a new flow group

                # get the new stroke group
                new_stroke = flow_group_counter

                # initliaze the flow group as an empty dictionary
                flow_group_dict[new_stroke] = []

                # get a length counter
                length_counter = 0

                # loop through stroke edge indices
                for idx, edge_index in enumerate(stroke_edge_indices):

                    # get the actual edge index
                    index = (edge_index[0], edge_index[1], 0)
                    
                    # add to the dictionary
                    flow_group_dict[new_stroke].append(index)

                    # get the length of the edge
                    length_counter += length_edges_list[idx]

                    # break the loop in the last iteration so that the flow group counter is not extended twice
                    # in case length_counter is greater than the average length at the end of loop
                    if idx + 1 == len(stroke_edge_indices):
                        break

                    # if the length is greater than the average length then create a new flow group
                    if length_counter > average_length:
                        # increse the global flow group counter
                        flow_group_counter += 1

                        # get the new stroke group
                        new_stroke = flow_group_counter

                        # initliaze the flow group as an empty dictionary
                        flow_group_dict[new_stroke] = []

                        # reset the length counter
                        length_counter = 0

                # increase the flow group counter
                flow_group_counter += 1

    edges_gdf['flow_group'] = np.ones(len(edges_gdf), dtype=np.int64)*(-1)

    # loop through dictionary
    for key in flow_group_dict.keys():
        # get the list of edges
        edge_list = flow_group_dict[key]

        # loop through the edges in the list
        for index in edge_list:
            # add the flow group to the dataframe
            edges_gdf.loc[index, 'flow_group'] = int(key)
    
    # find any flow groups less than merge_threshold
    groups_to_merge = []
    # loop through the flow_group_dict_keys
    for key in flow_group_dict.keys():

        # skip group 0 because we don't care about it
        if key == 0:
            continue

        # get the length of the flow group
        edges_in_flow_group = flow_group_dict[key]
        length_flow_group = edges_gdf.loc[edges_in_flow_group, 'length'].sum()

        # get the stroke group it belongs to
        stroke_group = edges_gdf.loc[edges_in_flow_group, 'stroke_group'].iloc[0]

        if length_flow_group <= merge_threshold:
            groups_to_merge.append((key, stroke_group))

    # TODO: small bug merge the groups
    for group in groups_to_merge:
        # get the flow group
        flow_group = group[0]

        # get the stroke group
        stroke_group = group[1]

        # get the edges in the flow group
        edges_in_flow_group = flow_group_dict[flow_group]

        # get the flow gdf
        flow_gdf = edges_gdf.loc[edges_gdf['flow_group'] == flow_group]

        # get the flow indices in order
        flow_edge_indices, length_edges_list = order_group_edges(flow_gdf)
        first_node_flow = flow_edge_indices[0][0]
        last_node_flow = flow_edge_indices[-1][1]

        # get the edges in the edge_gdf that belong to the stroke group
        stroke_gdf = edges_gdf.loc[edges_gdf['stroke_group'] == stroke_group]

        # get the stroke edge indices in order
        stroke_edge_indices, length_edges_list = order_group_edges(stroke_gdf)

        # convert stroke edge indices to a list
        nodes_in_stroke = np.array([item for t in stroke_edge_indices for item in t])

        # get a count of how many occurences each node has in the nodes_in_stroke numpy array
        v, c = np.unique(nodes_in_stroke, return_counts=True)

        # now check how many times the first_node_flow and last_node_flow occur in the nodes_in_stroke
        first_node_count = c[v == first_node_flow][0]
        last_node_count = c[v == last_node_flow][0]

        # if the first_node_count = 2 and last_node_count = 1 then the flow group is at the end of the stroke

        if first_node_count == 2 and last_node_count == 1:
            # check in stroke_edge_indices for the appeance of first_node_flow in the second poisition of the tuple
            edge_index = [x for x in stroke_edge_indices if first_node_flow == x[1]][0]
            preceding_flow_group = edges_gdf.loc[edge_index, 'flow_group'].iloc[0]

            # merge the flow groups
            flow_group_dict[preceding_flow_group] = flow_group_dict[preceding_flow_group] + flow_group_dict[flow_group]

        # if the first_node_count = 1 and last_node_count = 2 then the flow group is at the start of the stroke
        elif first_node_count == 1 and last_node_count == 2:
            # check in stroke_edge_indices for the appeance of the last_node_flow in the first poisition of the tuple
            edge_index = [x for x in stroke_edge_indices if last_node_flow == x[0]][0]
            following_flow_group = edges_gdf.loc[edge_index, 'flow_group'].iloc[0]

            # merge the flow groups
            flow_group_dict[following_flow_group] = flow_group_dict[following_flow_group] + flow_group_dict[flow_group]
        
        # if the first_node_count = 2 and last_node_count = 2 then the flow group is in the middle of the stroke
        elif first_node_count == 2 and last_node_count == 2:
            # check in stroke_edge_indices for the appeance of first_node_flow in the second poisition of the tuple
            # check in stroke_edge_indices for the appeance of the last_node_flow in the first poisition of the tuple
            # get both lengths of these flow groups and merge the current to the smaller one
            edge_index = [x for x in stroke_edge_indices if first_node_flow == x[1]][0]
            preceding_flow_group = edges_gdf.loc[edge_index, 'flow_group'].iloc[0]

            edge_index = [x for x in stroke_edge_indices if last_node_flow == x[0]][0]
            following_flow_group = edges_gdf.loc[edge_index, 'flow_group'].iloc[0]

            # check the lengths of the preceding and following flow groups
            length_preceding_flow_group = edges_gdf.loc[flow_group_dict[preceding_flow_group], 'length'].sum()
            length_following_flow_group = edges_gdf.loc[flow_group_dict[following_flow_group], 'length'].sum()

            # merge the flow groups
            if length_preceding_flow_group < length_following_flow_group:
                flow_group_dict[preceding_flow_group] = flow_group_dict[preceding_flow_group] + flow_group_dict[flow_group]
            else:
                flow_group_dict[following_flow_group] = flow_group_dict[following_flow_group] + flow_group_dict[flow_group]

    # loop through dictionary again to get the new flow groups
    # to ensure they are in order restart the count of the keys
    # TODO: for loop could fail if it doesn't start with zero
    new_key = 0
    for key in flow_group_dict.keys():

        if key == 0:
            if not new_key == 0:
                print('FIX THE CODE!!!, see todo above')

        # get the list of edges
        edge_list = flow_group_dict[key]

        # loop through the edges in the list
        for index in edge_list:
            # add the flow group to the dataframe
            edges_gdf.loc[index, 'flow_group'] = new_key
        
        # increment the key
        new_key += 1

    return edges_gdf

def find_split_indices(stroke_edge_indices, length_edges_list, approximate_length, num_splits, average_length):
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

    if num_edges > 3:

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

        # get the split choice
        best_split_choice_index = list(poss_split_indices[min_index])

    else:
        # when there are only three edges there is only one  split location
        split_choices_error = []
        # set the approximate length
        for split_loc in possible_split_locations:
            # get the sum of the possible splits
            front_sum = sum(length_edges_list[:split_loc])
            back_sum = sum(length_edges_list[split_loc:])

            # front and back error
            front_error = abs(front_sum - average_length)
            back_error = abs(back_sum - average_length)

            # sum the error
            split_choice_error = front_error + back_error
            
            # append the error
            split_choices_error.append(split_choice_error)

        # get the minimum split choice
        min_split_choice = min(split_choices_error)
        min_index = split_choices_error.index(min_split_choice)

        # get the split choice
        best_split_choice_index = possible_split_locations[min_index] 

    return best_split_choice_index

def order_group_edges(stroke_gdf):
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
