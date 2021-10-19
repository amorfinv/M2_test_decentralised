# %%
import osmnx as ox
import geopandas as gpd
from os import path
import numpy as np
from time import time
# use osmnx environment here

'''
Prepare graph for height allocation genetic algorithm.
'''

def main():

    # working path
    gis_data_path = 'gis'

    # Load MultiDigraph from create_graph.py
    G = ox.load_graphml(filepath=path.join(gis_data_path, 'streets', 'prep_height_allocation.graphml'))

    # convert to gdf
    nodes, edges = ox.graph_to_gdfs(G)

    # initiaize cost estimate
    init_genome, group_dict, node_connectivity, stroke_lenghts = init_height_cost_estimate(nodes, edges)

    # calculate cost
    cost = cost_estimate(init_genome, group_dict, node_connectivity, stroke_lenghts)

def init_height_cost_estimate(nodes_gdf, edges_gdf):
    """Initialize parameters for the genetic algorithm.

    Args:
        nodes_gdf (geopandas.GeoDataFrame): gdf of nodes
        edges_gdf (geopandas.GeoDataFrame): gdf of edges with stroke group and their length

    Returns:
        genome (numpy.array): numpy array containing genome. 0 is "height 0" and 1 is "height 1". 
                              The index of the array is group number,
        group_dict (dict): Contains the edge as the key and stroke group as the value,
        node_connectivity (dict): Contains the node as the key and the edges connecting them as the value
        stroke_lenghts (numpy.array): numpy array contatining stroke lenghts.
                                      The index of the array is the group number
    """

    # stroke_group array
    stroke_group_array = np.sort(np.unique(edges_gdf.loc[:,'stroke_group']).astype(np.int16))

    # create dictionary of edges with group num
    group_dict = {index:np.int16(row['stroke_group']) for index, row in edges_gdf.iterrows()}

    # create the genome
    genome = []
    stroke_lenghts = []
    for group in stroke_group_array:
        
        # create group_gdf
        group_gdf = edges_gdf.loc[edges_gdf['stroke_group'] == str(group)]

        # choose first edge in group and get layer allocation    
        layer_allocation = group_gdf.loc[group_gdf.index[0], 'layer_allocation']

        if layer_allocation == 'height 0':
            genome.append(0)
        else:
            genome.append(1)

        # get length of stroke_group
        stroke_lenghts.append(np.sum(group_gdf['length']))
    genome = np.array(genome, dtype=np.int8)

    # get list of node_ids and edge_ids
    node_osmids = nodes_gdf.index.values
    edge_uv = edges_gdf.index.values

    # create dictionary with node as key and connected edges as value
    node_connectivity = {}

    for osmid in node_osmids:
        
        # edges with node
        edges_with_node = [item for item in edge_uv if osmid in item]

        node_connectivity[osmid] = edges_with_node

    # get cost with current info
    group_dict = {index:np.int16(row['stroke_group']) for index, row in edges_gdf.iterrows()}
    
    return genome, group_dict, node_connectivity, stroke_lenghts

def cost_estimate(genome, group_dict, node_connectivity, stroke_lenghts):
    """Function to calculate the cost of the layer arrangement.

    Args:
        genome (numpy.array): numpy array containing genome. 0 is "height 0" and 1 is "height 1". 
                              The index of the array is group number,
        group_dict (dict): Contains the edge as the key and stroke group as the value,
        node_connectivity (dict): Contains the node as the key and the edges connecting them as the value
        stroke_lenghts (numpy.array): numpy array contatining stroke lenghts.
                                      The index of the array is the group number
    Returns:
        [float]: Returns the cost of the layer arrangement
    """
    genome = np.array(genome)
    
    # cost of bad degree-2 node
    cost_deg_2 = 0.5

    # cost of bad degree-3 node
    cost_deg_3 = 1

    # cost of bad degree-4 node
    cost_deg_4 = 2
    cost_deg_4_funky = 1.5

    # cost of bad degree-5 node
    cost_deg_5 = 2

    # cost of bad degree-6 node
    cost_deg_6 = 2

    # initialize cost to zero
    cost = 0

    # loop through each node 
    for osmid, edges_with_node in node_connectivity.items():

        # start looping through dict here and calculate cost at each node
        edges_with_node = node_connectivity[osmid]

        # get node degree
        node_degree = len(edges_with_node)

        # get unique groups at a node and put into a list
        groups_at_node = list({group_dict[edge] for edge in edges_with_node})
        group_sets = len(groups_at_node)

        if node_degree == 2:
            # cost estimate with degree-2 nodes

            # In a degree-2 intersections, there can be two groups so there are two layer heights
            
            # first check if there is one group
            if group_sets == 1:
                print(f'Warning: Degree-2 node {osmid} has one group')
                continue
                
            # If there are two different groups then we don't hurt the genome so much 
            # but still incur a cost for being in different group
            elif group_sets == 2:
                layer_first = genome[groups_at_node[0]]
                layer_second = genome[groups_at_node[1]]

                length_first = stroke_lenghts[groups_at_node[0]]
                length_second = stroke_lenghts[groups_at_node[1]]
                
                if layer_first != layer_second:
                    cost += cost_deg_2*(length_first*length_second)
                else:
                    # no cost if they are on dame
                    # print(f'No cost at node {osmid}')
                    continue             

        elif node_degree == 3:
            # cost estimate with degree-3 nodes

            # There should always be two groups in a degree-3 node. If not there is something strange
            if group_sets == 2:
                # check layer allocation of both
                layer_first = genome[groups_at_node[0]]
                layer_second = genome[groups_at_node[1]]

                length_first = stroke_lenghts[groups_at_node[0]]
                length_second = stroke_lenghts[groups_at_node[1]]

                # Incur a cost if the layer allocations are the same
                if layer_first == layer_second:
                    cost += cost_deg_3*(length_first*length_second)
                else:
                    # no cost if they are of different groups
                    # print(f'No cost at node {osmid}')
                    continue
            else:
                print(f'Check degree-3 node {osmid}. It appears there are strange groupings')
            
        elif node_degree == 4:

            # cost estimate with degree-4 nodes
            if group_sets == 1:
                print(f'Warning: Funky stuff at degree-4 node {osmid}. It appears there is just one group')
            elif group_sets == 2:
                # check layer allocation of both
                layer_first = genome[groups_at_node[0]]
                layer_second = genome[groups_at_node[1]]

                length_first = stroke_lenghts[groups_at_node[0]]
                length_second = stroke_lenghts[groups_at_node[1]]

                # Incur a cost if the layer allocations are the same
                if layer_first == layer_second:
                    cost += cost_deg_4*(length_first*length_second)

            elif group_sets == 3:
                # check the layer allocation of all three and put into a set and make a list
                layer_first = genome[groups_at_node[0]]
                layer_second = genome[groups_at_node[1]]
                layer_third = genome[groups_at_node[2]]

                length_first = stroke_lenghts[groups_at_node[0]]
                length_second = stroke_lenghts[groups_at_node[1]]
                length_third = stroke_lenghts[groups_at_node[2]]
                
                # Get the cost of the ones in the same layer height
                if layer_first == layer_second == layer_third:
                    cost += cost_deg_4_funky*(length_first*length_second*length_third)

                elif layer_first == layer_second:
                    cost += cost_deg_4_funky*(length_first*length_second)

                elif layer_first == layer_third:
                    cost += cost_deg_4_funky*(length_first*length_third)
                
                elif layer_second == layer_third:
                    cost += cost_deg_4_funky*(length_second*length_third)

            elif group_sets == 4:
                print(f'Warning: Funky stuff at degree-4 node {osmid}. It appears there are 4 groups')

            
        elif node_degree == 5:
            # cost estimate with degree-5 nodes

            if group_sets == 3:
                layer_first = genome[groups_at_node[0]]
                layer_second = genome[groups_at_node[1]]
                layer_third = genome[groups_at_node[2]]

                length_first = stroke_lenghts[groups_at_node[0]]
                length_second = stroke_lenghts[groups_at_node[1]]
                length_third = stroke_lenghts[groups_at_node[2]]

                # Get the cost of the ones in the same layer height
                if layer_first == layer_second == layer_third:
                    cost += cost_deg_5*(length_first*length_second*length_third)

                elif layer_first == layer_second:
                    cost += cost_deg_5*(length_first*length_second)

                elif layer_first == layer_third:
                    cost += cost_deg_5*(length_first*length_third)
                
                elif layer_second == layer_third:
                    cost += cost_deg_5*(length_second*length_third)
            elif group_sets == 4:
                layer_first = genome[groups_at_node[0]]
                layer_second = genome[groups_at_node[1]]
                layer_third = genome[groups_at_node[2]]
                layer_fourth = genome[groups_at_node[3]]

                length_first = stroke_lenghts[groups_at_node[0]]
                length_second = stroke_lenghts[groups_at_node[1]]
                length_third = stroke_lenghts[groups_at_node[2]]
                length_fourth = stroke_lenghts[groups_at_node[2]]

                # make arrays
                length_array = np.array([length_first, length_second, length_third, length_fourth])
                layer_array = np.array([layer_first, layer_second, layer_third, layer_fourth])

                bool_array_0 = (np.where(layer_array==0, True, False))
                bool_array_1 = (np.where(layer_array==1, True, False))
                
                cost_0 = length_array[bool_array_0]
                cost_1 = length_array[bool_array_1]

                if len(cost_0) == 1:
                    cost += cost_deg_5*(np.prod(cost_1))
                elif len(cost_1) == 1:
                    cost += cost_deg_5*(np.prod(cost_0))
                else:
                    cost += cost_deg_5*(np.prod(cost_0) + np.prod(cost_1))
            else:
                print(f'Warning: funky stuff at degree-5 node {osmid}')

        elif node_degree == 6:
            
            if group_sets == 4:
                layer_first = genome[groups_at_node[0]]
                layer_second = genome[groups_at_node[1]]
                layer_third = genome[groups_at_node[2]]
                layer_fourth = genome[groups_at_node[3]]

                length_first = stroke_lenghts[groups_at_node[0]]
                length_second = stroke_lenghts[groups_at_node[1]]
                length_third = stroke_lenghts[groups_at_node[2]]
                length_fourth = stroke_lenghts[groups_at_node[2]]

                # make arrays
                length_array = np.array([length_first, length_second, length_third, length_fourth])
                layer_array = np.array([layer_first, layer_second, layer_third, layer_fourth])

                bool_array_0 = (np.where(layer_array==0, True, False))
                bool_array_1 = (np.where(layer_array==1, True, False))
                
                cost_0 = length_array[bool_array_0]
                cost_1 = length_array[bool_array_1]

                if len(cost_0) == 1:
                    cost += cost_deg_6*(np.prod(cost_1))
                elif len(cost_1) == 1:
                    cost += cost_deg_6*(np.prod(cost_0))
                else:
                    cost += cost_deg_6*(np.prod(cost_0) + np.prod(cost_1))
            else:
                print(f'Warning: Not accounting for cost in degree-6 node {osmid}')

        else:
            print(f'There are no costs for a degree-{node_degree} node')

    return float(cost),


if __name__ == '__main__':
    main()
    