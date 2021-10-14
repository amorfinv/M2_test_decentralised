from enum import unique
from hashlib import new
from platform import node
import osmnx as ox
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely import ops
import math
import geopandas as gpd
from pyproj import CRS
import ast
from collections import Counter
from shapely.geometry.multipoint import MultiPoint
import rtree
from shapely.strtree import STRtree
from shapely.geometry import Point
from shapely import affinity
import pandas as pd

def new_edge_osmid_to_point(nodes, edges, osmid, point, group):
    # intersect edge or at a specific point
    
    # create copies
    nodes_gdf = nodes.copy()
    edges_gdf = edges.copy()

    # get osmid geometry
    osmid_geom = nodes_gdf.loc[osmid, 'geometry']

    # create a linestring
    line_new = LineString([osmid_geom, point])

    # put edges in a tree
    geom_list = []
    for index, row in edges_gdf.iterrows():
        
        geom = row.loc['geometry']
        geom.id = index

        geom_list.append(geom)

    # add geometry to an RTREE
    tree = STRtree(geom_list)

    # see where new line intersects
    potential_intersections = tree.query(line_new)

    # no intersection
    no_intersects = True

    for potential_intersection in potential_intersections:
        
        # remove ones that do not intersect new line
        if potential_intersection.intersects(line_new) and osmid not in potential_intersection.id:
        
            intersecting_point = potential_intersection.intersection(line_new)

            # Split the intersected edge
            edge_to_split = potential_intersection.id
            new_node_osmid = get_new_node_osmid(nodes_gdf)
            split_loc = intersecting_point

            node_new, row_new1, row_new2 = split_edge(edge_to_split, new_node_osmid, nodes_gdf, edges_gdf, split_loc)

            # append nodes and edges and remove split edge
            nodes_gdf = nodes_gdf.append(node_new)
            edges_gdf = edges_gdf.append([row_new1, row_new2])
            edges_gdf.drop(index=edge_to_split, inplace=True)

            # Now add a new edge between the nodes
            new_edge = (osmid, new_node_osmid, 0)
            edges_gdf = edges_gdf.append(new_edge_straight(new_edge, nodes_gdf, edges_gdf, group))

            no_intersects = False

    
    if no_intersects:
        # if no intersections just create a new node and edge at location

        # create a new osmid
        new_node_osmid = get_new_node_osmid(nodes_gdf)

        # create a new node and add to node_gdf
        nodes_gdf = create_new_node(nodes_gdf, point, new_node_osmid)

        # name the new edge
        new_edge = (osmid, new_node_osmid, 0)

        # add new edge to gdf
        edges_gdf = edges_gdf.append(new_edge_straight(new_edge, nodes_gdf, edges_gdf, group))

    return nodes_gdf, edges_gdf

def create_new_node(nodes, geom, node_osmid):

    nodes_gdf = nodes.copy()

    #round new point geometry
    new_node_x = round(geom.x, 7)
    new_node_y = round(geom.y, 7)

    # recreate point
    new_node_geom = Point([new_node_x, new_node_y])

    # get dummy dictionary of node to keep same attributes in new node
    node_dict = nodes.iloc[0].to_dict()

    # update data of new node_id.
    node_dict["x"] = new_node_geom.x
    node_dict["y"] = new_node_geom.y
    node_dict["geometry"] = [new_node_geom]
    node_dict["osmid"] = node_osmid
    
    if 'street_count' in node_dict:
        node_dict["street_count"] = 4

    # get new node gdf
    node_new = gpd.GeoDataFrame(node_dict, crs=nodes.crs)
    node_new.set_index(['osmid'], inplace=True)

    nodes_gdf = nodes_gdf.append(node_new)

    return nodes_gdf

def extend_edges(nodes, edges, edge_list):

    # create nodes
    nodes_gdf = nodes.copy()
    edges_gdf = edges.copy()

    # create graph
    G = ox.graph_from_gdfs(nodes_gdf, edges_gdf)

    # get edge uvs
    edges_uv = list(edges_gdf.index)
    # edge_curr = edge_list[1]

    for edge_curr in edge_list:

        # get list of geometries and add edge attribute as its id and create tree (can be outside for loop probably)
        geom_list = []
        for index, row in edges_gdf.iterrows():
            
            geom = row.loc['geometry']
            geom.id = index

            geom_list.append(geom)

        # add geometry to an RTREE
        tree = STRtree(geom_list)

        # get geometry
        geom_line = edges_gdf.loc[edge_curr, 'geometry']

        # Find which node needs to be extended. front (u) or back (v)
        deg_node_u = G.degree[edge_curr[0]]
        deg_node_v = G.degree[edge_curr[1]]

        if deg_node_u == 1:
            node_extend = edge_curr[0]
            node_ignore = edge_curr[1]

            idx_line = 'front'

        elif deg_node_v == 1:
            node_extend = edge_curr[1]
            node_ignore = edge_curr[0]

            idx_line = 'back'

        # while loop should start here to extend line if necessary
        while True:

            # get potential intersections
            potential_intersections = tree.query(geom_line)

            actual_intersections = []
            for intersection in potential_intersections:
                # only check if current edge does not have node to ignore
                if not node_ignore in intersection.id:
                    
                    # check if there is an actual intersection or not
                    temp_geom = edges_gdf.loc[intersection.id, 'geometry']

                    if temp_geom.intersects(geom_line):
                        split_point = geom_line.intersection(temp_geom)
                        actual_intersections.append((intersection.id, split_point))

            # check if you need to extend the line. This is true if actual_intersections is empty
            if not actual_intersections:

                # get coordinates into a list
                coords = list(geom_line.coords)

                # extend line from front of edge geometry
                if idx_line == 'front':
                    extension_coords = coords[:2]
                    line_to_extend = LineString(extension_coords)
                    
                    new_start = list(affinity.scale(line_to_extend, xfact=2, yfact=2, zfact=2, origin=extension_coords[1]).coords)[0]

                    new_line = LineString([new_start, coords[1]])

                # extend line from back of edge geometry
                elif idx_line == 'back':
                    extension_coords = coords[-2:]
                    line_to_extend = LineString(extension_coords)

                    new_end = list(affinity.scale(line_to_extend, xfact=2, yfact=2, zfact=2, origin=extension_coords[0]).coords)[-1]

                    new_line = LineString([coords[0], new_end])

                # assign new value to geom_line
                geom_line = new_line

            else:

                # If there is an intersection ensure that there is just one. otherwise code it up
                if len(actual_intersections) > 1:
                    print(f'There is more than one intersection when extending {edge_curr}')
                    print('Recode stuff!')
                    break
                
                # Now edit the gdfs

                # Split the intersected edge
                edge_to_split = actual_intersections[0][0]
                new_node_osmid = get_new_node_osmid(nodes_gdf)
                split_loc = actual_intersections[0][1]

                # split edge
                node_new, row_new1, row_new2 = split_edge(edge_to_split, new_node_osmid, nodes_gdf, edges_gdf, split_loc)

                # append nodes and edges and remove split edge
                nodes_gdf = nodes_gdf.append(node_new)
                edges_gdf = edges_gdf.append([row_new1, row_new2])
                edges_gdf.drop(index=edge_to_split, inplace=True)

                # Now add a new edge between the nodes
                if idx_line == 'front':
                    new_edge = (new_node_osmid, node_extend, 0)
                elif idx_line == 'back':
                    new_edge = (node_extend, new_node_osmid, 0)
                
                group = edges_gdf.loc[edge_curr, 'stroke_group']
                edges_gdf = edges_gdf.append(new_edge_straight(new_edge, nodes_gdf, edges_gdf, group))

                break



    
    return nodes_gdf, edges_gdf

def node_degree_attrib(nodes, edges):

    # build graph
    G = ox.graph_from_gdfs(nodes, edges)
    
    # get degree of nodes
    node_degree = [G.degree[osmid] for osmid in nodes.index.values]

    return node_degree

def get_first_group_edges(G, edges):
    """[summary]

    Args:
        G ([type]): [description]
        edges ([type]): [description]

    Returns:
        [type]: [description]
    """    
    edgedict = dict()
    first_edges = []
    
    # Create own directionary of edges
    for index, edge in edges.iterrows():
        group_no = edge['stroke_group']
        if group_no not in edgedict:
            edgedict[group_no] = []
        edgedict[group_no].append(index)

    # If a node only appears once, then that edge is either the first or last of a group. unless it is a circular group
    for group_no in edgedict:
        group_edges = np.array(edgedict[group_no])
        flat_edges = group_edges.flatten()
        flat_edges_nonzero = flat_edges[flat_edges != 0]
        values, counts = np.unique(flat_edges_nonzero, return_counts = True)
        unique_index = np.argwhere(counts == 1)

        # check for circuitous groups
        circuitous_group = np.argwhere(counts>2)
        if len(circuitous_group):
           print(f'Group {group_no} is circuitous') 

        if len(unique_index):
            first_node = values[unique_index[0][0]]
        else:
            # set unique index to zero for circular groups
            first_node = values[0]
            print(f'Group {group_no} is circular')
        for edge in group_edges:
            if edge[0] == first_node:
                first_edges.append((edge[0], edge[1], edge[2]))
                break
            
            elif edge[1] == first_node:
                first_edges.append((edge[1], edge[0], edge[2]))
                break
        
    return first_edges

def consolidate_unconnected_nodes(nodes, edges, nodes_to_consolidate, new_node='centroid', new_node_coords=None, node_loc='extend'):
    """ 
    Function takes a list of unconnected nodes and consolidates them into one node. 
    It also takes all edges that unconnected nodes were a part of and connects them to the new node.
    You can either take the centroid of the unconnected nodes as the location for the new node or
    give an (x,y) coordinate. At the moment the edge geometries get extended from the position of 
    the unconnectednode.

    TODO: future idea is to create a consolidate_connected_nodes

    Args:
        nodes (gdf): node geodataframe
        edges (gdf): edge geodataframe
        nodes_to_consolidate (list): List of nodes that will be consolidated.
                                       Nodes must be in order to form a normal polygon
        new_node (str, optional): Choose the location of the new node. It can either be 'centroid'
                                  or 'custom'. If 'custom', then new_node_coords should have x,y.
                                  Defaults to 'centroid'.
        new_node_coords (list, optional): The desired location of the new node. Should be a list 
                                          with [x,y] positions. Defaults to None.
        node_loc (str, optional): Decides if the edge geometry is extended from nodes_to_consolidate
                                  or if the point is replaced with the new node location. At the moment
                                  it is only possible to have 'extend'. TODO: add functionality 
                                  for 'replace'. Defaults to 'extend'.

    Returns:
        nodes_gdf (gdf): Returns the updated node geodataframe with the removed consolidated nodes and
                         added new node.
        edges_gdf (gdf): Returns the updated edge geodataframe where edges now contain the new node
                         instead of the consolidated nodes.

    """    
    # create copy of gdfs
    nodes_gdf = nodes.copy()
    edges_gdf = edges.copy()

    # get location of the new node to replace nodes_to consolidate
    # there are a couple options planned. One is the centroid, or a custom value
    if new_node == 'centroid':
        
        point_list = []
        for _node in nodes_to_consolidate:

            point_geom = nodes_gdf.loc[_node,'geometry']
            point_list.append(point_geom)
        
        # get centroid of new point
        new_node_point = Polygon(point_list).centroid

    elif new_node == 'custom':
        new_node_point = Point(new_node_coords)

    # get x, y values of centroid and round
    new_node_x = round(new_node_point.x, 7)
    new_node_y = round(new_node_point.y, 7)

    # # recreate point
    new_node_point = Point([new_node_x, new_node_y])

    # get max and min node id so we can generate a new node id.
    # node_osmids = nodes_gdf.index.values
    # max_node_id = np.amax(node_osmids)
    # min_node_id = np.amin(node_osmids)

    # if min_node_id - 1 > 0: 
    #     new_node_id = min_node_id - 1
    # else:
    #     new_node_id = max_node_id + 1
    
    new_node_id = get_new_node_osmid(nodes_gdf)

    # get dummy dictionary to keep same attributes
    node_dict = nodes_gdf.loc[nodes_to_consolidate[0]].to_dict()
    
    # update data of new node_id.
    node_dict["x"] = new_node_x
    node_dict["y"] = new_node_y
    node_dict["geometry"] = [new_node_point]
    node_dict["osmid"] = new_node_id
    
    if 'street_count' in node_dict:
        node_dict["street_count"] = 4

    # get nodes to delete from node_gdf and delete
    nodes_to_delete = nodes_to_consolidate
    nodes_gdf.drop(index=nodes_to_delete, axis=0, inplace=True)

    # add new node to gdf
    node_gdf_new = gpd.GeoDataFrame(node_dict, crs=edges.crs)
    node_gdf_new.set_index(['osmid'], inplace=True)
    nodes_gdf = nodes_gdf.append(node_gdf_new)

    # Now that nodes gdf has been fixed.. update the edge_gdf
    edge_uv = edges_gdf.index.tolist()

    # find edges with nodes_to_delete and get information
    for idx, node_to_replace in enumerate(nodes_to_delete):
        node_to_replace = nodes_to_delete[idx]
        edge_to_edit = [item for item in edge_uv if node_to_replace in item][0] # fix this when doing nodes that are connected


        # get geoseries of that row and convert to dictionary
        row_df = edges_gdf.loc[edge_to_edit]
        row_dict = row_df.to_dict()
        line_geom = row_dict['geometry']
        
        # see which side to append geometry
        node_pos = 0 if node_to_replace == edge_to_edit[0]  else -1

        # TODO: node_loc extend or replace. Right now it just extends linestrings
        if node_loc == 'extend':

            if node_pos == 0:
                # create new segment with new point and first point of original line
                new_segment = LineString([new_node_point, line_geom.coords[node_pos]])

                # combine lines into one
                multi_line = MultiLineString([new_segment, line_geom])
                merged_line = ops.linemerge(multi_line)

                new_edge = (new_node_id, edge_to_edit[1], edge_to_edit[2])

            elif node_pos == -1:
                # create new segment with new point and first point of original line
                new_segment = LineString([line_geom.coords[node_pos], new_node_point])

                # combine lines into one
                multi_line = MultiLineString([line_geom, new_segment])
                merged_line = ops.linemerge(multi_line)

                new_edge = (edge_to_edit[0], new_node_id, edge_to_edit[2])
        

        # change the geometry attribute in dictionary with new merged line
        row_dict['geometry'] = [merged_line]

        # check if length attribute is part of dictionary and update geometry
        if 'length' in row_dict:
            row_dict['length'] = round(merged_line.length * (111000), 2) # convert from degrees to meters. TODO: reprojecting is better
        
        row_dict['u'] = new_edge[0]
        row_dict['v'] = new_edge[1]
        row_dict['key'] = 0

        # create gdf of new edge and append to edges_gdf
        edge_gdf_new = gpd.GeoDataFrame(row_dict, crs=edges.crs)
        edge_gdf_new.set_index(['u', 'v', 'key'], inplace=True)
        edges_gdf = edges_gdf.append(edge_gdf_new)

        # drop old gdf
        edges_gdf.drop(index=edge_to_edit, inplace=True)

    return nodes_gdf, edges_gdf

def initial_prep(G):
    '''
    Argument is an osmnx Graph.

    Clean graph by removing some attributes.
    TODO: maybe we don't need 'one_way' in edges if remove_two_way_edges or remove_long_way_edges is 
        adapted.

    Returns an osmnx Graph
    
    '''
    # convert to geodataframe for easier manipulation
    nodes, edges = ox.graph_to_gdfs(G)

    nodes_clean = nodes.filter(['y','x','street_count','geometry'], axis=1)
    edges_clean = edges.filter(['oneway', 'length', 'geometry'])

    # re-build graph
    G_clean = ox.graph_from_gdfs(nodes_clean, edges_clean)

    return G_clean

def remove_unconnected_streets(G):
    '''
    Argument is an osmnx Graph.

    Removes unconnected streets (dead-ends) from graph. 
    It searches for street_count = 1 inside the node geodataframe and removes them from graph. 
    Not the same as removing degree-1 nodes. Degree-1 nodes may have a street count higher than 1

    Returns an osmnx Graph

    TODO: perhaps faster to do with networX.
    
    '''
    # convert to geodataframe for easier manipulation
    nodes, _ = ox.graph_to_gdfs(G)

    # select list of osmid nodes to remove
    node_ids = nodes[nodes['street_count'] == 1].index.to_list()

    # remove nodes from Graph
    G.remove_nodes_from(node_ids)

    return G

def allocate_edge_height(edges_gdf, rotation_val=0):
    '''
    Arguments:
        -edges_gdf: edge geodataframe from osmnx. The edges geodataframe must contain a column with bearings.
        -rotation_val: rotation of bounds (uniform). Must be between -45° and 45°.

    Add layer height allocation attribute to graph based on street bearing.

          high_bound_1    N    low_bound_1
            ░░░░░\░░░░░░░░│░░░░░░░░/░░░░░
            ░░░░░░\░░░░░░░│░░░░░░░/░░░░░░
            ░░░░░░░\░░░░░░│░░░░░░/░░░░░░░
            ░░░░░░░░\░░░░░│░░░░░/░░░░░░░░
            ░░░░░░░░░\░░░░│░░░░/░░░░░░░░░
            ░░░░░░░░░░\░░░│░░░/░░░░░░░░░░
            ░░░░░░░░░░░\░░│░░/░╬░░░░░░░░░
            ░░░░░░░░░░░░\░│░/░░░╬----------rotation angle (min: -45° and max:45°)
            ░░░░░░░░░░░░░\│/░░░░░╬░░░░░░░
           W──────────────┼──────────────E
            ░░░░░░░░░░░░░/│\░░░░░░░░░░░░░
            ░░░░░░░░░░░░/░│░\░░░░░░░░░░░░
            ░░░░░░░░░░░/░░│░░\░░░░░░░░░░░
            ░░░░░░░░░░/░░░│░░░\░░░░░░░░░░
            ░░░░░░░░░/░░░░│░░░░\░░░░░░░░░
            ░░░░░░░░/░░░░░│░░░░░\░░░░░░░░
            ░░░░░░░/░░░░░░│░░░░░░\░░░░░░░
            ░░░░░░/░░░░░░░│░░░░░░░\░░░░░░
            ░░░░░/░░░░░░░░│░░░░░░░░\░░░░░
          low_bound_2     S    high_bound_1

    Shows border between height 1 and height 2 layers. For example if rotation angle is zero:
        -low_bound_1 = 45°
        -high_bound_1 = 135°
        -low_bound_2 = 225°
        -high_bound_2 = 315°
    This means that any street with bearing angle between 45° and 135° or 225° and 315° will be at height 1. All others will be at height 2.
    In this function the rotation is uniform, meaning that rotation angle is added to all bounds. Usually, you want to remove edges that
    are not very straight before passing it onto this method. This is because osmnx only gives bearings from node to node.

    It is not recommended to use this method for a large city-wide graph. However, it will be useful to optimize subset graphs.
    
    Returns geodataframes of edges. These can be used to create a new graph outisde function.

    TODO: Make it so that arbitrary rotation angles can be applied. 
    At the moment a uniform rotation to an orhtogonal axis is used for cutoff. However, there is no reason why this needs to be orthogonal.
    This new addition will make optimization more difficult and perhaps not very necessary. This method is meant to be an initial
    optimization. After this optimization, some refinement is needed to fix some edges.
    Exception handling. So if rotation angle is too small/large then choose a correct value.
    '''

    # get borders to divide layer allocation (see image in comments)
    low_bound_1 = 45 + rotation_val
    high_bound_1 = 135 + rotation_val
    low_bound_2 = 225 + rotation_val
    high_bound_2 = 315 + rotation_val

    # create numpy array of bearing values, initialize layer allocation column for dataframe and allocate layer heights
    bearings = edges_gdf['bearing'].to_numpy()
    layer_allocation = []

    for bearing in np.nditer(bearings):

        if low_bound_1 < bearing < high_bound_1 or low_bound_2 < bearing < high_bound_2:
            layer_loc = 'height 1'
        else:
            layer_loc = 'height 2'

        # allocate layers
        layer_allocation.append(layer_loc)

    # get bearing difference index. Not using bearing indices for now
    # bearing_indices = calculate_bearing_diff_indices(bearings, low_bound_1, high_bound_1, low_bound_2, high_bound_2)
    
    return layer_allocation

def allocate_group_height(nodes_gdf, edges_gdf, rotation_val=0):
    '''
    Arguments:
        -nodes_gdf: node gdf containing information about x,y locations.
        -edges_gdf: edge geodataframe with group numbers.
        -rotation_val: rotation of bounds (uniform). Must be between -45° and 45°.

    Add layer height allocation attribute to graph based on street bearing.

          high_bound_1    N    low_bound_1
            ░░░░░\░░░░░░░░│░░░░░░░░/░░░░░
            ░░░░░░\░░░░░░░│░░░░░░░/░░░░░░
            ░░░░░░░\░░░░░░│░░░░░░/░░░░░░░
            ░░░░░░░░\░░░░░│░░░░░/░░░░░░░░
            ░░░░░░░░░\░░░░│░░░░/░░░░░░░░░
            ░░░░░░░░░░\░░░│░░░/░░░░░░░░░░
            ░░░░░░░░░░░\░░│░░/░╬░░░░░░░░░
            ░░░░░░░░░░░░\░│░/░░░╬----------rotation angle (min: -45° and max:45°)
            ░░░░░░░░░░░░░\│/░░░░░╬░░░░░░░
           W──────────────┼──────────────E
            ░░░░░░░░░░░░░/│\░░░░░░░░░░░░░
            ░░░░░░░░░░░░/░│░\░░░░░░░░░░░░
            ░░░░░░░░░░░/░░│░░\░░░░░░░░░░░
            ░░░░░░░░░░/░░░│░░░\░░░░░░░░░░
            ░░░░░░░░░/░░░░│░░░░\░░░░░░░░░
            ░░░░░░░░/░░░░░│░░░░░\░░░░░░░░
            ░░░░░░░/░░░░░░│░░░░░░\░░░░░░░
            ░░░░░░/░░░░░░░│░░░░░░░\░░░░░░
            ░░░░░/░░░░░░░░│░░░░░░░░\░░░░░
          low_bound_2     S    high_bound_1

    Angles start at east and go counter-clockwise

    Shows border between height 1 and height 2 layers. For example if rotation angle is zero:
        -low_bound_1 = 45°
        -high_bound_1 = 135°
        -low_bound_2 = 225°
        -high_bound_2 = 315°
    This means that any street with bearing angle between 45° and 135° or 225° and 315° will be at height 1. All others will be at height 2.
    In this function the rotation is uniform, meaning that rotation angle is added to all bounds. Usually, you want to remove edges that
    are not very straight before passing it onto this method. This is because osmnx only gives bearings from node to node.

    It is not recommended to use this method for a large city-wide graph. However, it will be useful to optimize sub graphs.
    
    Returns layer_height allocation gdf that can be added to the original gdf
    '''

    # get borders to divide layer allocation (see image in comments)
    low_bound_1 = 45 + rotation_val
    high_bound_1 = 135 + rotation_val
    low_bound_2 = 225 + rotation_val
    high_bound_2 = 315 + rotation_val

    # get numpy array of group numbers
    group_numbers = np.unique(edges_gdf['stroke_group'])

    # See which height each group belongs to
    group_heights = {}
    for group_num in group_numbers:

        # get start and end node
        start_node, end_node = get_group_border_nodes(group_num, edges_gdf)

        orig = (nodes_gdf.loc[start_node].y, nodes_gdf.loc[start_node].x)
        dest = (nodes_gdf.loc[end_node].y, nodes_gdf.loc[end_node].x)

        # get group bearing (from first to last node) and shift coordinate system of osmnx funciton
        group_bearing = get_bearing(orig, dest)
        if 0 <= group_bearing <= 90:
            group_bearing = 90 - group_bearing
        elif 90 < group_bearing <= 360:
            group_bearing = 360 - (group_bearing - 90)
        
        # allocate layer depending on bounds
        if low_bound_1 < group_bearing < high_bound_1 or low_bound_2 < group_bearing < high_bound_2:
            layer_loc = 'height 1'
        else:
            layer_loc = 'height 2'

        group_heights[group_num] = layer_loc
    
    # get layer_allocation
    layer_allocation = edges_gdf['stroke_group'].apply(lambda group_num: group_heights[group_num])

    return layer_allocation

def get_group_border_nodes(group_num, edges_gdf):

    '''
    Given a group number and the edge gdf the function gives the id of the first and last node
    of the directed graph.
    TODO: consolidate into one while loop
    '''
    
    # get a gdf of just the stroke group and just it's values
    group_gdf = edges_gdf[edges_gdf['stroke_group']==group_num]
    group_uv = list(group_gdf.index.values)

    # get initial start and end node from first edge
    idx = 0
    start_node = group_uv[idx][0]
    group_uv_start = group_uv[:idx] + group_uv[idx + 1:]
    start_flag =  True
    while start_flag:
        
        try:
            edges_with_start_node = [item for item in group_uv_start if start_node in item][0]

            start_node = edges_with_start_node[0]
        
            idx = group_uv_start.index(edges_with_start_node)
            group_uv_start = group_uv_start[:idx] + group_uv_start[idx + 1:]
        
        except IndexError:
            start_flag = False

    idx = 0
    end_node = group_uv[idx][1]
    group_uv_end = group_uv[:idx] + group_uv[idx + 1:]
    end_flag = True
    while end_flag:
        
        try:
            edges_with_end_node = [item for item in group_uv_end if end_node in item][0]
            end_node = edges_with_end_node[1]
    
            idx = group_uv_end.index(edges_with_end_node)
            group_uv_end = group_uv_end[:idx] + group_uv_end[idx + 1:]

            xx = False
        except IndexError:
            end_flag = False

    return start_node, end_node

def calculate_integral_bearing_difference(edges_geometry):
    '''
    Argument:
        -edge_geometry: numpy array containing the LineString information of each edge

    The function integrates the bearing change across each edge. This is used later to remove edges not part 
    of rotation angle optimization. See allocate_edge_height for more information.

    Returns a numpy array containing the integral bearing differences which can then be added to the edge geodataframe.

    TODO: check if integration takes signs into account correctly. clockwise and anticlockwise changes should be 
    accounted for with opposite signs. so a weaving street may get an integral bearing difference of zero.
    Does not seem to be taken into account
    '''

    # initialize variable of for loop to put integration result of each edge
    integral_bearings_diff = []

    for _, line_string in np.ndenumerate(edges_geometry):

        # unpack lat lon values from shapely linestring. First value is lon, second is lat
        lon, lat = np.array(line_string.coords.xy)

        if not len(lat) == 2:
            # only go into loop if there are more than 2 edges in the linestring (or 3 sets of points)
            n = len(lat) -1

            # initialize array that captures segment bearings
            bearing_list = np.empty(n)

            for jdx in range(0, n):
                # get origin and destination point of edge jdx. there are N + 1 edges and N points
                origin_point = (lat[jdx], lon[jdx])
                destination_point = (lat[jdx + 1], lon[jdx + 1])

                # Get bearing from osmnx function and save to array
                bearing_list[jdx] = get_bearing(origin_point, destination_point)
            
            # get bearing difference of first and last segment. Same as summing the difference (this is the integral difference)
            integral_difference = abs(bearing_list[-1] - bearing_list[0])
            integral_difference = 360 - integral_difference if integral_difference > 180 else integral_difference


            # append to list with absolute value since we don't care about direction
            integral_bearings_diff.append(round(abs(integral_difference)))

        else:
            # set integral bearing difference to zero when only one edge or 2 sets of points
            integral_bearings_diff.append(0)

    return integral_bearings_diff

def get_bearing(origin_point, destination_point):
    """
    Code obtained from osmnx library. See notes below.
    ---------------------------------------------------------------------

    Calculate the bearing between two lat-lng points.

    Each argument tuple should represent (lat, lng) as decimal degrees.
    Bearing represents angle in degrees (clockwise) between north and the
    direction from the origin point to the destination point.

    Parameters
    ----------
    origin_point : tuple
        (lat, lng)
    destination_point : tuple
        (lat, lng)

    Returns
    -------
    bearing : float
        the compass bearing in decimal degrees from the origin point to the
        destination point
    """
    if not (isinstance(origin_point, tuple) and isinstance(destination_point, tuple)):
        raise TypeError("origin_point and destination_point must be (lat, lng) tuples")

    # get latitudes and the difference in longitude, as radians
    lat1 = np.radians(origin_point[0])
    lat2 = np.radians(destination_point[0])
    diff_lng = np.radians(destination_point[1] - origin_point[1])

    # calculate initial bearing from -180 degrees to +180 degrees
    x = np.sin(diff_lng) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(diff_lng))
    initial_bearing = np.arctan2(x, y)

    # normalize initial bearing to 0-360 degrees to get compass bearing
    initial_bearing = np.degrees(initial_bearing)
    bearing = initial_bearing % 360

    return bearing

def calculate_bearing_diff_indices(bearings, low_bound_1, high_bound_1, low_bound_2, high_bound_2):
    '''
    Arguments:
        -bearings: numpy array with all street bearings.
        -low_bound_1, high_bound_1, low_bound_2, high_bound_2: boundary angles defined in allocate_edge_height method.
    Code used to calculate the bearing difference index.

    This is an index that calculates the average bearing difference of all streets in their allocated height for allocated 
    height 1, height 2, and for their summation.

    Returns the bearing indices for height 1, height 2, and their summation.

    TODO: perhaps it is better to just calculate the variance. And maybe it is more statistically significant and easier.
    Maybe there is a way to combine the three into one index.

    '''
    # seperate bearing arrays into respective heights
    height_1_array = bearings[(bearings > low_bound_1) & (bearings < high_bound_1) | (bearings > low_bound_2) & (bearings < high_bound_2)]
    height_2_array = np.setdiff1d(bearings, height_1_array)

    # subtract 180 degrees from all values larger than 180
    height_1_array = np.where(height_1_array > 180, height_1_array - 180, height_1_array)
    height_2_array = np.where(height_2_array > 180, height_2_array - 180, height_2_array)

    # combine into a tuple
    height_arrays = (height_1_array, height_2_array)

    # initialize empty tuple
    diff_index = (np.empty(height_1_array.size), np.empty(height_2_array.size))

    for jdx in range(0, len(height_arrays)):
        height_array = height_arrays[jdx]

        for idx, bearing in np.ndenumerate(height_array):

            avg_difference = np.sum(abs(bearing - height_array)) / (height_array.size - 1)
            diff_index[jdx][idx] = avg_difference
        
    # get bearing difference index
    index_1 = diff_index[0].mean()
    index_2 = diff_index[1].mean()
    index_combined = index_1 + index_2

    return index_1, index_2, index_combined

def add_edge_interior_angles(edges):
    '''
    Function creates an interior angle function.

    For one edge, function finds all intersecting edges and gets the angles between them using COINS
    '''
    
    # create dataframe with u,v and linestring geometry
    geo = edges["geometry"]

    # Create list of tuples with indices from edge geodataframe (u, v, 0)
    edge_uv = list(geo.index.values)
    
    # Initialize list
    edge_interior_angle = []

    for edge_current in edge_uv:

        #select nodes to check
        node_1 = edge_current[0]
        node_2 = edge_current[1]

        # get geometry of edge (first line segment and last line segments) and combine into a list. First line segment starts from node 1. 
        # Second line segment starts from node 2. This is done because there can be several points in a linestring. However it also works with 2 point line.
        geom = geo.loc[edge_current]
        line_segment_node_1 = list(map(LineString, zip(geom.coords[:-1], geom.coords[1:])))[0]
        line_segment_node_1 = list(line_segment_node_1.coords)

        line_segment_node_2 = list(map(LineString, zip(geom.coords[:-1], geom.coords[1:])))[-1]
        line_segment_node_2 = list(line_segment_node_2.coords)

        # combine line segments of edge
        line_segment_list = [line_segment_node_1, line_segment_node_2]

        # Get all intersecting edges and remove current edge
        edges_with_nodes = [item for item in edge_uv if node_1 in item or node_2 in item]
        edges_with_nodes.remove(edge_current)

        # initialize edge dictionary that will contain interior angle between line segments.
        # Perhaps check previous calculations to see if angle has already been calculated and just copy
        edge_dict = {}
        for edge in edges_with_nodes:
            # get linestring geometry of edge
            geom1 = geo.loc[edge]

            # check if node is in u,v position. If u, get first segment of linestring (idx=0). If v, get last segment of linestring (idx=-1)
            idx = 0 if node_1 == edge[0] or node_2 == edge[0] else -1

            # check to see which line segement to use from line_segment_list. It depends which node is in the current edge
            idc = 0 if node_1 in edge else 1

            # Create line segment of intersecting edge and convert to list
            line_segment = list(map(LineString, zip(geom1.coords[:-1], geom1.coords[1:])))[idx]
            line_segment = list(line_segment.coords)

            # get angle
            angle = angleBetweenTwoLines(line_segment_list[idc], line_segment)
            
            # save to dictionary
            edge_dict[edge] = angle
        
        edge_interior_angle.append(edge_dict)

    return edge_interior_angle

def angleBetweenTwoLines(line1, line2):
    """
    The below function calculates the joining angle between
    two line segments. 
    source: https://github.com/PratyushTripathy/COINS
    """

    l1p1, l1p2 = line1
    l2p1, l2p2 = line2
    l1orien = computeOrientation(line1)
    l2orien = computeOrientation(line2)
    """
    If both lines have same orientation, return 180
    If one of the lines is zero, exception for that
    If both the lines are on same side of the horizontal plane, calculate 180-(sumOfOrientation)
    If both the lines are on same side of the vertical plane, calculate pointSetAngle
    """
    if (l1orien==l2orien): 
        angle = 180
    elif (l1orien==0) or (l2orien==0): 
        angle = pointsSetAngle(line1, line2)
        
    elif l1p1 == l2p1:
        if ((l1p1[1] > l1p2[1]) and (l1p1[1] > l2p2[1])) or ((l1p1[1] < l1p2[1]) and (l1p1[1] < l2p2[1])):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = pointsSetAngle([l1p1, l1p2], [l2p1,l2p2])
    elif l1p1 == l2p2:
        if ((l1p1[1] > l2p1[1]) and (l1p1[1] > l1p2[1])) or ((l1p1[1] < l2p1[1]) and (l1p1[1] < l1p2[1])):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = pointsSetAngle([l1p1, l1p2], [l2p2,l2p1])
    elif l1p2 == l2p1:
        if ((l1p2[1] > l1p1[1]) and (l1p2[1] > l2p2[1])) or ((l1p2[1] < l1p1[1]) and (l1p2[1] < l2p2[1])):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = pointsSetAngle([l1p2, l1p1], [l2p1,l2p2])
    elif l1p2 == l2p2:
        if ((l1p2[1] > l1p1[1]) and (l1p2[1] > l2p1[1])) or ((l1p2[1] < l1p1[1]) and (l1p2[1] < l2p1[1])):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = pointsSetAngle([l1p2, l1p1], [l2p2,l2p1])
    return(angle)

def pointsSetAngle(line1, line2):
    """
    This below function calculates the acute joining angle between
    two given set of points.
    source: https://github.com/PratyushTripathy/COINS
    """
    
    l1orien = computeOrientation(line1)
    l2orien = computeOrientation(line2)
    if ((l1orien>0) and (l2orien<0)) or ((l1orien<0) and (l2orien>0)):
        return(abs(l1orien)+abs(l2orien))
    elif ((l1orien>0) and (l2orien>0)) or ((l1orien<0) and (l2orien<0)):
        theta1 = abs(l1orien) + 180 - abs(l2orien)
        theta2 = abs(l2orien) + 180 - abs(l1orien)
        if theta1 < theta2:
            return(theta1)
        else:
            return(theta2)
    elif (l1orien==0) or (l2orien==0):
        if l1orien<0:
            return(180-abs(l1orien))
        elif l2orien<0:
            return(180-abs(l2orien))
        else:
            return(180 - (abs(computeOrientation(line1)) + abs(computeOrientation(line2))))
    elif (l1orien==l2orien):
        return(180)

def computeOrientation(line):
    '''
    Compute orientation of a line
    source: https://github.com/PratyushTripathy/COINS
    '''
    point1 = line[1]
    point2 = line[0]
    """
    If the latutide of a point is less and the longitude is more, or
    If the latitude of a point is more and the longitude is less, then
    the point is oriented leftward and wil have negative orientation.
    """
    if ((point2[0] > point1[0]) and (point2[1] < point1[1])) or ((point2[0] < point1[0]) and (point2[1] > point1[1])):
        return(-computeAngle(point1, point2))
    #If the latitudes are same, the line is horizontal
    elif point2[1] == point1[1]:
        return(0)
    #If the longitudes are same, the line is vertical
    elif point2[0] == point1[0]:
        return(90)
    else:
        return(computeAngle(point1, point2))

def computeAngle(point1, point2):

    """
    The function below calculates the angle between two points in space.
    source: https://github.com/PratyushTripathy/COINS
    """

    height = abs(point2[1] - point1[1])
    base = abs(point2[0] - point1[0])
    angle = round(math.degrees(math.atan(height/base)), 3)
    return(angle)

def remove_two_way_edges(edges):

    '''
    argument: edges (gdf)
    Function removes one edge from any two-way edges. The removed edge is arbitrary.

    returns edge gdf
    '''

    # create copy of gdf
    edges_gdf = edges.copy()

    # create sub geodataframe with only oneway==False streets
    one_way_bool = edges_gdf[edges_gdf["oneway"] == False]["oneway"]

    # isolate u,v components
    edge_indices = list(one_way_bool.index.values)

    # initialize list that will contain edges to be removed
    edges_remove = []

    # while loop it will check all edges that are duplicated and remove them
    while edge_indices:

        # Choose first edge. Since we are removing edge at end of loop, we always check first edge
        edge_to_check = edge_indices[0]

        # create possible duplicate edge
        edge_duplicate = (edge_to_check[1], edge_to_check[0], 0)

        # If duplicat edge is in the list of indices then add it the remove list.
        if edge_duplicate in edge_indices:
            edges_remove.append(edge_to_check)
        
        # Delete the edge to check from list before restarting loop
        edge_indices.remove(edge_to_check)
    
    # remove all duplicate edges from geodataframe
    edges_gdf.drop(edges_remove, inplace=True)

    # set one-way column to true. Not really needed but just for quality of life
    edges_gdf["oneway"].values[:] =  True

    return edges_gdf

def remove_long_way_edges(edges):

    '''
    inputs an edge_gdf

                ░░░░░░░░░░░░░░
                ░░┌────────█░░ Node 1
                ░░│░░░░░░░░│░░
                ░░│░░░░░░░░│░░
                ░░│░░░░░░░░│░░
        edge 2  ░░│░░░░░░░░│░░ edge 1
                ░░│░░░░░░░░│░░
                ░░│░░░░░░░░│░░
                ░░│░░░░░░░░│░░
                ░░└────────█░░ Node 2
                ░░░░░░░░░░░░░░
    Possible configuration A
        eddge 1 = (Node 1, Node 2, 0)
        eddge 2 = (Node 1, Node 2, 1)
    Possible configuration B
        eddge 1 = (Node 1, Node 2, 0)
        eddge 2 = (Node 2, Node 1, 0)
    
    Configurations come from osmnx standards
    
    function removes edge 2 as it is the long way edge between two nodes no matter the directionality
    TODO: combine with remove_two_way_edges
    '''

    # create copy of edge_gdf
    edges_gdf = edges.copy()

    # create sub geodataframe with only oneway== True streets
    one_way_bool = edges_gdf[edges_gdf["oneway"] == True]["oneway"]

    # isolate u,v components
    edge_indices = list(one_way_bool.index.values)

    # initiaite list that will contain edges to be removed
    edges_remove = []

    # while loop it will check all edges that are duplicate and remove
    while edge_indices:

        # Choose first edge. Since we are removing edge at end of loop, we always check first edge
        edge_to_check = edge_indices[0]

        # create possible duplicate edge
        edge_duplicates = [ (edge_to_check[0], edge_to_check[1], 1), (edge_to_check[1], edge_to_check[0], 0)]

        edge_selected = [i for i in edge_duplicates if i in edge_indices[1:]]

        # If duplicat edge is in the list of indices then add it the remove list.
        if edge_selected:
            # get length of mirror edges to remove longer length
            length_to_check = edges_gdf.loc[edge_to_check, 'length']
            length_sel = edges_gdf.loc[edge_selected[0], 'length']
            
            # remove edge with longer length and add to list
            edge_to_remove = edge_to_check if length_to_check > length_sel else edge_selected[0]
            edges_remove.append(edge_to_remove)
        
        # Delete the edge to check from list before restarting loop
        edge_indices.remove(edge_to_check)

    # remove from geodataframe
    edges_gdf.drop(edges_remove, inplace=True)

    return edges_gdf

def edge_naming_simplification(edges):

    '''
    inputs an edge_gdf

                ░░░░░░░░░░░░░░
                ░░┌────────█░░ Node 1
                ░░│░░░░░░░░│░░
                ░░│░░░░░░░░│░░
                ░░│░░░░░░░░│░░
                ░░│░░░░░░░░│░░
                ░░│░░░░░░░░│░░
                ░░│░░░░░░░░│░░
                ░░│░░░░░░░░│░░
                ░░└────────█░░ Node 2
                ░░░░░░░░░░░░░░
    Possible naming configurations
        edge = (Node 1, Node 2, 0)
        edge = (Node 1, Node 2, 1)
        edge = (Node 2, Node 1, 0)
        edge = (Node 1, Node 2, 1)

    '''

    # create copy of edge_gdf
    edges_gdf = edges.copy()

    # isolate u,v components of edge
    edge_indices = list(edges_gdf.index)

    # initiaite list that will contain edges to be removed
    edges_remove = []

    # while loop it will check all edges that are duplicate and remove
    while edge_indices:

        # Choose first edge. Since we are removing edge at end of loop, we always check first edge
        edge_to_check = edge_indices[0]
        edge_remove = edge_to_check

        # create possible duplicate edges
        edge_duplicates = [(edge_to_check[0], edge_to_check[1], 0), 
                           (edge_to_check[0], edge_to_check[1], 1),
                           (edge_to_check[1], edge_to_check[0], 0),
                           (edge_to_check[1], edge_to_check[0], 1)]

        edge_selected = [i for i in edge_duplicates if i in edge_indices[1:]]

        # If duplicat edge is in the list of indices then add it to the remove list
        if edge_selected:    
            # get list with all possible edge choices
            edge_choices = [edge_to_check] + edge_selected

            # get length of edges in a list
            length_edges = [edges_gdf.loc[edge, 'length'] for edge in edge_choices]
            
            # get index of largest balue
            index_to_choose = length_edges.index(max(length_edges))

            # new edge to add to gdf
            # get dict of selected edge and set key=0 and fix geometry
            row_dict = edges.loc[edge_choices[index_to_choose]].to_dict()
            row_dict['u'] = edge_choices[index_to_choose][0]
            row_dict['v'] = edge_choices[index_to_choose][1]
            row_dict['key'] = 0
            row_dict['geometry'] = [row_dict['geometry']]

            # create gdf of new edge and append to edges_gdf
            row_new = gpd.GeoDataFrame(row_dict, crs=edges.crs)
            row_new.set_index(['u', 'v', 'key'], inplace=True)

            # now remove all edge choices from gdf and add new choice
            edges_gdf.drop(index=edge_choices, inplace=True)          
            edges_gdf = edges_gdf.append(row_new)

            # remove relevant edges from edge_indices
            for edge in edge_choices[1:]:
                edge_indices.remove(edge)

            # update edge to check so it doesn't go into next loop
            edge_to_check = (edge_choices[index_to_choose][0], edge_choices[index_to_choose][1], 0)
            

        # Change edge keys that have a 1 in the key value but have no duplicates
        if edge_to_check[2] == 1:

            # new edge to add to gdf
            # get dict of selected edge and set key=0 and fix geometry
            row_dict = edges.loc[edge_to_check].to_dict()
            row_dict['u'] = edge_to_check[0]
            row_dict['v'] = edge_to_check[1]
            row_dict['key'] = 0
            row_dict['geometry'] = [row_dict['geometry']]

            # create gdf of new edge and append to edges_gdf
            row_new = gpd.GeoDataFrame(row_dict, crs=edges.crs)
            row_new.set_index(['u', 'v', 'key'], inplace=True)

            # now remove all edge choices from gdf and add new choice
            edges_gdf.drop(index=edge_to_check, inplace=True)          
            edges_gdf = edges_gdf.append(row_new)
        
        # Delete the edge to check from list before restarting loop
        edge_indices.remove(edge_remove)

    return edges_gdf

def set_direction(nodes, edges, edge_directions):

    # Create list of stroke groups
    stroke_groups = list(np.sort(np.unique(np.array(edges['stroke_group'].values))))
    
    # We need to put edge_directions in the same order as the edges
    new_edge_directions = []
    for num_group, stroke_group_list in enumerate(stroke_groups):
        # get edges in specific group
        edge_in_group = edges.loc[edges['stroke_group']== stroke_group_list]

        # get edge indices of stroke group in a list
        edge_uv = list(edge_in_group.index.values)
        for direction in edge_directions:
            if (direction in edge_uv) or ((direction[1], direction[0], 0) in edge_uv):
                new_edge_directions.append(direction)
                break
            
    edge_directions = new_edge_directions

    # initialize correct index order list and line data list for new geo dataframe
    index_order = []
    my_geodata = []

    for num_group, stroke_group_list in enumerate(stroke_groups):

        # get edges in specific group
        edge_in_group = edges.loc[edges['stroke_group']== stroke_group_list]

        # get edge indices of stroke group in a list
        edge_uv = list(edge_in_group.index.values)

        # get desired group direction from edge_directions
        group_direction = edge_directions[num_group]

        # Find index of direction setting edge
        if group_direction in edge_uv:
            jdx = edge_uv.index(group_direction)
        else:
            jdx = edge_uv.index((group_direction[1], group_direction[0], 0))

        # create counter for while loop
        idx = 0

        # create a copy of edge_in_group for while loop. TODO: smarter way to do this
        edges_removed = edge_in_group
        numb_edges_stroke = len(edge_in_group)
        search_direct_front = True  # true if searching from front
        search_direct_back = True
        while len(edges_removed):
            curr_edge = edge_in_group.iloc[jdx]
            curr_index = edge_uv[jdx]

            if curr_index[0] == group_direction[0]:
                #print(f'{curr_index} edge going correct direction')
                new_index = (curr_index[0], curr_index[1], 0)
                edge_line_direct = edge_in_group.loc[curr_index, 'geometry']

            else:
                #print(f'{curr_index} edge going incorrect direction')
                new_index = (curr_index[1], curr_index[0], 0)
                
                # reverse linestring from edge
                wrong_line_direct = list(edge_in_group.loc[curr_index, 'geometry'].coords)
                wrong_line_direct.reverse()
                edge_line_direct = LineString(wrong_line_direct)
                
            # see what columns to add in new gdf
            dummy_edge_dict = edges_removed.loc[curr_index].to_dict()

            column_names = []
            for key in dummy_edge_dict:
                column_names.append(key)

            # drop edge from dataframe for while loop (TODO: smarter way to do this)
            edges_removed = edges_removed.drop(index=curr_index)

            # add new_index to index order
            index_order.append(new_index)

            # for loop to create sub geo data list
            sub_geo_data_list = [new_index[0], new_index[1], new_index[2]]
            for attr_key in column_names:
                # add geometry to dictionary
                if attr_key == 'geometry':
                    value = edge_line_direct
                
                # add length to dictionary if needed. TODO: not neccesary?
                elif attr_key == 'length':
                    value = round(edge_line_direct.length * (111000), 2)

                # add bearing if needed
                elif attr_key == 'bearing':
                    u_geom = list(nodes.loc[new_index[0], 'geometry'].coords)
                    v_geom = list(nodes.loc[new_index[1], 'geometry'].coords)
                    value = round(get_bearing(u_geom[0][::-1], v_geom[0][::-1]), 2)
                else:
                    value = edge_in_group.loc[curr_index, attr_key]
                sub_geo_data_list.append(value)
            
            column_names = ['u', 'v', 'key'] + column_names
            my_geodata.append(sub_geo_data_list)

            # add line_string data to list (make this dynamic)
            # oneway = edge_in_group.loc[curr_index, 'oneway']
            # length = edge_in_group.loc[curr_index, 'length']
            # geom = edge_line_direct
            # bearing = edge_in_group.loc[curr_index, 'bearing']
            # edge_interior_angle = edge_in_group.loc[curr_index, 'edge_interior_angle']
            # stroke_group_label = edge_in_group.loc[curr_index, 'stroke_group']

            # my_geodata.append([new_index[0], new_index[1], new_index[2],oneway, length, geom, bearing, edge_interior_angle, stroke_group_label])

            # set new jdx based on current edge (only if not last edge)
            if not idx == numb_edges_stroke - 1:
                # front node
                node_to_find_front = new_index[1]
                edge_with_node_front = [item for item in edge_uv if node_to_find_front in item]

                # back node
                node_to_find_back = new_index[0]
                edge_with_node_back = [item for item in edge_uv if node_to_find_back in item]

                # either go forwards or backwards
                edge_with_node_front.remove(curr_index)
                edge_with_node_back.remove(curr_index)

                if (edge_with_node_front and search_direct_front):
                    search_direct_back = False
                    next_edge = edge_with_node_front[0]
                    jdx = edge_uv.index(next_edge)

                    # set desired direction for next edge
                    if node_to_find_front == next_edge[0]:
                        group_direction = next_edge
                    else:
                        group_direction = (next_edge[1], next_edge[0], 0)
                
                elif (edge_with_node_back and search_direct_back):
                    search_direct_front = False
                    next_edge = edge_with_node_back[0]
                    jdx = edge_uv.index(next_edge)

                    # set desired direction for next edge
                    if node_to_find_back == next_edge[1]:
                        group_direction = next_edge
                    else:
                        group_direction = (next_edge[1], next_edge[0], 0)
            
            # advance counter
            idx = idx + 1

    # create edge geodataframe
    # column_names = ['u', 'v', 'key', 'oneway', 'length', 'geometry', 'bearing', 'edge_interior_angle', 'stroke_group']
    edge_gdf = gpd.GeoDataFrame(my_geodata, columns=column_names, crs=edges.crs)

    edge_gdf.set_index(['u', 'v', 'key'], inplace=True)

    return edge_gdf

def set_direction_group(nodes, edges, edge_direction):

    # Create list of stroke groups
    try:
        stroke_group = edges.loc[edge_direction, 'stroke_group']
    except:
        stroke_group = edges.loc[(edge_direction[1], edge_direction[0], 0), 'stroke_group']

    
    edge_directions = edge_direction

    # initialize correct index order list and line data list for new geo dataframe
    index_order = []
    my_geodata = []

    # get edges in specific group
    edge_in_group = edges.loc[edges['stroke_group']== stroke_group]

    # get edge indices of stroke group in a list
    edge_uv = list(edge_in_group.index.values)

    # get desired group direction from edge_directions
    group_direction = edge_directions

    # Find index of direction setting edge
    if group_direction in edge_uv:
        jdx = edge_uv.index(group_direction)
    else:
        jdx = edge_uv.index((group_direction[1], group_direction[0], 0))

    # create counter for while loop
    idx = 0

    # create a copy of edge_in_group for while loop. TODO: smarter way to do this
    edges_removed = edge_in_group
    numb_edges_stroke = len(edge_in_group)
    search_direct_front = True  # true if searching from front
    search_direct_back = True
    while len(edges_removed):
        curr_edge = edge_in_group.iloc[jdx]
        curr_index = edge_uv[jdx]

        if curr_index[0] == group_direction[0]:
            #print(f'{curr_index} edge going correct direction')
            new_index = (curr_index[0], curr_index[1], 0)
            edge_line_direct = edge_in_group.loc[curr_index, 'geometry']

        else:
            #print(f'{curr_index} edge going incorrect direction')
            new_index = (curr_index[1], curr_index[0], 0)
            
            # reverse linestring from edge
            wrong_line_direct = list(edge_in_group.loc[curr_index, 'geometry'].coords)
            wrong_line_direct.reverse()
            edge_line_direct = LineString(wrong_line_direct)
            
        # see what columns to add in new gdf
        dummy_edge_dict = edges_removed.loc[curr_index].to_dict()

        column_names = []
        for key in dummy_edge_dict:
            column_names.append(key)

        # drop edge from dataframe for while loop (TODO: smarter way to do this)
        edges_removed = edges_removed.drop(index=curr_index)

        # add new_index to index order
        index_order.append(new_index)

        # for loop to create sub geo data list
        sub_geo_data_list = [new_index[0], new_index[1], new_index[2]]
        for attr_key in column_names:
            # add geometry to dictionary
            if attr_key == 'geometry':
                value = edge_line_direct
            
            # add length to dictionary if needed. TODO: not neccesary?
            elif attr_key == 'length':
                value = round(edge_line_direct.length * (111000), 2)

            # add bearing if needed
            elif attr_key == 'bearing':
                u_geom = list(nodes.loc[new_index[0], 'geometry'].coords)
                v_geom = list(nodes.loc[new_index[1], 'geometry'].coords)
                value = round(get_bearing(u_geom[0][::-1], v_geom[0][::-1]), 2)
            else:
                value = edge_in_group.loc[curr_index, attr_key]
            sub_geo_data_list.append(value)
        
        column_names = ['u', 'v', 'key'] + column_names
        my_geodata.append(sub_geo_data_list)

        # set new jdx based on current edge (only if not last edge)
        if not idx == numb_edges_stroke - 1:
            # front node
            node_to_find_front = new_index[1]
            edge_with_node_front = [item for item in edge_uv if node_to_find_front in item]

            # back node
            node_to_find_back = new_index[0]
            edge_with_node_back = [item for item in edge_uv if node_to_find_back in item]

            # either go forwards or backwards
            edge_with_node_front.remove(curr_index)
            edge_with_node_back.remove(curr_index)

            if (edge_with_node_front and search_direct_front):
                search_direct_back = False
                next_edge = edge_with_node_front[0]
                jdx = edge_uv.index(next_edge)

                # set desired direction for next edge
                if node_to_find_front == next_edge[0]:
                    group_direction = next_edge
                else:
                    group_direction = (next_edge[1], next_edge[0], 0)
            
            elif (edge_with_node_back and search_direct_back):
                search_direct_front = False
                next_edge = edge_with_node_back[0]
                jdx = edge_uv.index(next_edge)

                # set desired direction for next edge
                if node_to_find_back == next_edge[1]:
                    group_direction = next_edge
                else:
                    group_direction = (next_edge[1], next_edge[0], 0)
        
        # advance counter
        idx = idx + 1

    # remove old edges
    edges.drop(index=edge_in_group.index, inplace=True)
    edge_gdf = gpd.GeoDataFrame(my_geodata, columns=column_names, crs=edges.crs)
    edge_gdf.set_index(['u', 'v', 'key'], inplace=True)

    edges_new = edges.append(edge_gdf)

    return edges_new

def edge_gdf_format_from_gpkg(edges):

    # check if geometry is a multilinestring and convert
    if edges.geometry.type[0] == 'MultiLineString':
        edges['geometry'] = edges['geometry'].apply(lambda multi_line: ops.linemerge(multi_line))

    edge_dict = edges.to_dict()

    edge_gdf = gpd.GeoDataFrame(edge_dict, crs=CRS.from_user_input(4326))
    edge_gdf.set_index(['u', 'v', 'key'], inplace=True)

    return edge_gdf

def node_gdf_format_from_gpkg(nodes):

    # check if geometry is a multipoint and convert
    if nodes.geometry.type[0] == 'MultiPoint':
        nodes['geometry'] = nodes['geometry'].apply(lambda multi_point: Point(multi_point[0]))

    node_dict = nodes.to_dict()
    node_gdf = gpd.GeoDataFrame(node_dict, crs=CRS.from_user_input(4326))
    node_gdf.set_index(['osmid'], inplace=True)

    return node_gdf

def simplify_graph(nodes, edges, angle_cut_off = 120):
    '''
    Requires a COINS run before and a set_directions
    remove degree-2 edges with int angle greater than 120, does a fresh add_edge_interior angles
    Basically consolidates when deleting edges manually in qgis
    '''

    # add fresh interior angles
    edges['edge_interior_angle'] = add_edge_interior_angles(edges)

    # Make copies
    edges_gdf_new = edges.drop(['edge_interior_angle'], axis=1).copy()
    nodes_gdf_new = nodes.copy()

    # make a graph to check nodes of degree 2
    G_check = ox.graph_from_gdfs(nodes, edges)
    
    # get list if node ids and edge ids
    node_ids = list(nodes.index.values)
    edge_uv = list(edges.index.values)

    # initilaize a list
    nodes_to_delete = []
    edges_to_merge = []

    for osmid, node_deg in G_check.degree(node_ids):
        if node_deg == 2:
            # find edges with a degree-2 node
            edges_in_node = [item for item in edge_uv if osmid in item]

            edge_1 = edges_in_node[0]
            edge_2 = edges_in_node[1]

            # find interior angle between both edges
            int_angle_dict = edges.loc[edge_1, 'edge_interior_angle']
            int_angle = int_angle_dict[edge_2]

            # append edges in edges to merge if int_angle greater than angle cutoff
            if int_angle > angle_cut_off:
                nodes_to_delete.append(osmid)
                if edge_1 not in edges_to_merge:

                    edges_to_merge.append(edge_1)

                if edge_2 not in edges_to_merge:

                    edges_to_merge.append(edge_2)
    
    # run while there are nodes_to_delete
    while nodes_to_delete:

        node_inspect = nodes_to_delete[0]

        # create master edge
        edges_in_merge = [item for item in edges_to_merge if node_inspect in item]
        edge_1 = edges_in_merge[0]
        edge_2 = edges_in_merge[1]

        # get group number
        group_1 = edges_gdf_new.loc[edge_1, 'stroke_group']
        group_2 = edges_gdf_new.loc[edge_2, 'stroke_group']

        # ensure that it is an integer
        if isinstance(group_1, pd.Series):
            group_1 = group_1.values[0]
        
        if isinstance(group_2, pd.Series):
            group_2 = group_2.values[0]

        if group_1 != group_2:
            print(f'Cannot merge {edge_1} and {edge_2} because they are in different groups')
            nodes_to_delete.remove(node_inspect)
            continue

        # find which edge is in front and which is in back
        if node_inspect == edge_1[0]:
            edge_front = edge_1
            edge_back = edge_2
        else:
            edge_front = edge_2
            edge_back = edge_1

        # check if edge in back continues
        start_node = edge_back[0]
        start_edges_to_merge = [edge_back]

        while start_node in nodes_to_delete:
            node_inspect = start_node

            edges_in_merge1 = [item for item in edges_to_merge if node_inspect in item]
            edge_1 = edges_in_merge1[0]
            edge_2 = edges_in_merge1[1]

            if node_inspect == edge_1[0]:
                edge_back = edge_2
            else:
                edge_back = edge_1

            start_node = edge_back[0]
            start_edges_to_merge.append(edge_back)

        start_edges_to_merge = start_edges_to_merge[::-1]

        # check if edge in front continues
        end_node = edge_front[1]
        end_edges_to_merge = [edge_front]

        while end_node in nodes_to_delete:
            node_inspect = end_node

            edges_in_merge1 = [item for item in edges_to_merge if node_inspect in item]
            edge_1 = edges_in_merge1[0]
            edge_2 = edges_in_merge1[1]

            if node_inspect == edge_1[0]:
                edge_front = edge_1
            else:
                edge_front = edge_2

            end_node = edge_front[1]
            end_edges_to_merge.append(edge_front)
   
        # edges to merge
        merged_edges = start_edges_to_merge + end_edges_to_merge

        len_edges = len(merged_edges)
        nodes_pop = []
        # get list of nodes to remove in one go
        for idx in range(0, len_edges):

            if idx < len_edges - 1:
                node_pop = merged_edges[idx][1]
                nodes_pop.append(node_pop)

        
        # merge geometry here
        multi_line_list = []
        for edge in merged_edges:
            # merge line if interior angle is larger than 150 degrees
            line_string = edges.loc[edge , 'geometry']
            multi_line_list.append(line_string)


        multi_line = MultiLineString(multi_line_list)
        merged_line = ops.linemerge(multi_line)

        # get dummy dictionary to keep consistent columns
        row_dict = edges_gdf_new.iloc[0].to_dict()

        # fill dummy dictionary with correct values
        row_dict['geometry'] = [merged_line]

        # check if length or bearing attribute is part of dictionary and update geometry
        if 'length' in row_dict:
            row_dict['length'] = round(merged_line.length * (111000), 2) # convert from degrees to meters
        
        if 'bearing' in row_dict:
            u_geom = list(nodes_gdf_new.loc[merged_edges[0][0], 'geometry'].coords)
            v_geom = list(nodes_gdf_new.loc[merged_edges[-1][1], 'geometry'].coords)
            row_dict['bearing'] = round(get_bearing(u_geom[0][::-1], v_geom[0][::-1]), 2)

        row_dict['u'] = merged_edges[0][0]
        row_dict['v'] = merged_edges[-1][1]
        row_dict['key'] = 0
        
        row_new = gpd.GeoDataFrame(row_dict, crs=edges_gdf_new.crs)
        row_new.set_index(['u', 'v', 'key'], inplace=True)

        # append edge
        edges_gdf_new = edges_gdf_new.append(row_new)

        # delete edges from edge_gdf_new
        edges_gdf_new.drop(index=merged_edges, inplace=True)

        # delete node from nodes_gdf_new
        nodes_gdf_new.drop(index=nodes_pop, inplace=True)

        # remove
        for node_del in nodes_pop:
            nodes_to_delete.remove(node_del)

    # rebuild gdfs for quality of life
    nodes_gdf_new, edges_gdf_new = ox.graph_to_gdfs(ox.graph_from_gdfs(nodes_gdf_new, edges_gdf_new))

    # add edge interior angles
    # edges_gdf_new['edge_interior_angle'] = add_edge_interior_angles(edges_gdf_new)

    return nodes_gdf_new, edges_gdf_new

def new_groups_90(nodes, edges, angle_cut_off = 45):
    '''
    Create new groups for groups that turn 90 degrees +- 45
    '''
    edges_gdf_new = edges.copy()
    nodes_gdf_new = nodes.copy()

    G_check = ox.graph_from_gdfs(nodes, edges)
    
    node_ids = list(nodes.index.values)
    edge_uv = list(edges.index.values)

    nodes_to_split = []

    for osmid, node_deg in G_check.degree(node_ids):
        if node_deg == 2:
            # find edges with a degree-2 node
            edges_in_node = [item for item in edge_uv if osmid in item]

            edge_1 = edges_in_node[0]
            edge_2 = edges_in_node[1]

            # find interior angle between both edges (TODO: FIX how it reads rather than try/except)
            try:
                int_angle_dict = ast.literal_eval(edges.loc[edge_1, 'edge_interior_angle'])
            except ValueError:
                int_angle_dict = edges.loc[edge_1, 'edge_interior_angle']

            int_angle = int_angle_dict[edge_2]

            if  90 - angle_cut_off < int_angle < 90 + angle_cut_off :
                nodes_to_split.append(osmid)

    for node_split in nodes_to_split:
        
        # find group to split dynamically in case it has changed
        edges_with_node = [item for item in edge_uv if node_split in item]
        group_to_split_0 = edges_gdf_new.loc[edges_with_node[0], 'stroke_group']
        group_to_split_1 = edges_gdf_new.loc[edges_with_node[1], 'stroke_group']

        # Only split if the group number is the same
        if group_to_split_0 == group_to_split_1:

            # split group
            new_group_gdf = split_group_at_node(edges_gdf_new, node_split, group_to_split_0)
            new_group_edges = new_group_gdf.index.values

            # drop new_group dataframe new_group_gdf to edges_gdf
            edges_gdf_new.drop(index=new_group_edges, inplace=True)
            edges_gdf_new = edges_gdf_new.append(new_group_gdf)

    return nodes_gdf_new, edges_gdf_new

def connect_nodes(nodes, edges, nodes_to_connect, group=None):
    """
    Function takes a node and edge gdf and a list of edges to create in nodes_to_connect.
    All the edges in nodes_to_connect get added to the edges_gdf

    Args:
        nodes (GeoDataFrame): GeoDataFrame of nodes.
        edges (GeoDataFrame): GeoDataFrame of edges.
        nodes_to_connect (list): List of edges [(node1, node2), (node3, node4)....] to connect

    Returns:
        [GeoDataFrame]: An edge GeoDataFrame with the nodes_to_connect as new edges
    """    
    # make copy of nodes edges
    nodes_gdf = nodes.copy()
    edges_gdf= edges.copy()

    # loop through new edges and connect the nodes
    for new_edge in nodes_to_connect:
        edges_gdf = edges_gdf.append(new_edge_straight(new_edge, nodes_gdf, edges_gdf, group))

    return edges_gdf

def split_edge(edge_to_split, node_osmid, nodes, edges, split_loc=None, split_type='new'):
    '''

    function splits edge and returns new node geometry and new edges,
    at the moment delete edges outside of this function. At the moment it only works with linestrings
    cannot split a straight line yet

    split_loc is the index of a point in the linestring if split_type = 'idx'
    split_loc is a new Point if split_type='new'
    split_loc is a at an existing point not in linestring with split_type='exist'

    '''

    if split_type == 'new':

        # split edge at a new point in nodes gdf

        # get geometry of split edge       
        geom_split_main = edges.loc[edge_to_split, 'geometry']

        # round new point geometry
        new_node_x = round(split_loc.x, 7)
        new_node_y = round(split_loc.y, 7)

        # recreate point
        new_node_geom = Point([new_node_x, new_node_y])

        split_geom = split_line_with_point(geom_split_main, new_node_geom)

        geom_edge_1 = split_geom[0]
        geom_edge_2 = split_geom[1]

        # create new edges and new node
        new_edge_1 = (edge_to_split[0], node_osmid, edge_to_split[2])
        new_edge_2 = (node_osmid, edge_to_split[1], edge_to_split[2])

        # get dummy dictionary of node to keep same attributes in new node
        node_dict = nodes.loc[edge_to_split[0]].to_dict()

        # update data of new node_id.
        node_dict["x"] = new_node_geom.x
        node_dict["y"] = new_node_geom.y
        node_dict["geometry"] = [new_node_geom]
        node_dict["osmid"] = node_osmid
        
        if 'street_count' in node_dict:
            node_dict["street_count"] = 4

        # get new node gdf
        node_new = gpd.GeoDataFrame(node_dict, crs=edges.crs)
        node_new.set_index(['osmid'], inplace=True)


    elif split_type == 'idx':
        # split edge at an already existing point inside the linestring

        # get geometry of split edge
        split_geom = list(edges.loc[edge_to_split, 'geometry'].coords)

        # create new edges and new node
        new_edge_1 = (edge_to_split[0], node_osmid, edge_to_split[2])
        new_edge_2 = (node_osmid, edge_to_split[1], edge_to_split[2])

        # get new node and geom
        new_node_yx = split_geom[split_loc]
        new_node_geom = Point(new_node_yx)

        geom_edge_1 = LineString(split_geom[:split_loc + 1])
        geom_edge_2 = LineString(split_geom[split_loc:])

       # get dummy dictionary of node to keep same attributes in new node
        node_dict = nodes.loc[edge_to_split[0]].to_dict()

        # update data of new node_id.
        node_dict["x"] = new_node_geom.x
        node_dict["y"] = new_node_geom.y
        node_dict["geometry"] = [new_node_geom]
        node_dict["osmid"] = node_osmid
        
        if 'street_count' in node_dict:
            node_dict["street_count"] = 4

        # get new node gdf
        node_new = gpd.GeoDataFrame(node_dict, crs=edges.crs)
        node_new.set_index(['osmid'], inplace=True)


    elif split_type == 'exist':
        # split edge at an already existing point but that is not part of the linestring
        # get geometry of split edge       
        geom_split_main = edges.loc[edge_to_split, 'geometry']

        # round new point geometry
        new_node_x = round(split_loc.x, 7)
        new_node_y = round(split_loc.y, 7)

        # recreate point
        exist_node_geom = Point([new_node_x, new_node_y])

        split_geom = split_line_with_point(geom_split_main, exist_node_geom)

        geom_edge_1 = split_geom[0]
        geom_edge_2 = split_geom[1]

        # create new edges and new node
        new_edge_1 = (edge_to_split[0], node_osmid, edge_to_split[2])
        new_edge_2 = (node_osmid, edge_to_split[1], edge_to_split[2])

        # now check if new edges already exist and if it does add a value to key
        edges_text = [f'{edge[0]}-{edge[1]}-{edge[2]}' for edge in list(edges.index)]

        key_1 =0
        while f'{new_edge_1[0]}-{new_edge_1[1]}-{new_edge_1[2]}' in edges_text:
            key_1 += 1
            new_edge_1 = (new_edge_1[0], new_edge_1[1], key_1)

        key_2 =0
        while f'{new_edge_2[0]}-{new_edge_2[1]}-{new_edge_2[2]}' in edges_text:
            key_2 += 1
            new_edge_2 = (new_edge_2[0], new_edge_2[1], key_2)
    
    # create two dummy dicts of edges
    row_dict1 = edges.loc[edge_to_split].to_dict()
    row_dict2 = row_dict1.copy()

    # change the geometry attribute in dictionary with new merged line
    row_dict1['geometry'] = [geom_edge_1]

    # check if length attribute is part of dictionary and update geometry
    if 'length' in row_dict1:
        row_dict1['length'] = round(geom_edge_1.length * (111000), 2) # convert from degrees to meters
    
    row_dict1['u'] = new_edge_1[0]
    row_dict1['v'] = new_edge_1[1]
    row_dict1['key'] = new_edge_1[2]

    # create gdf of new edge and append to edges_gdf
    row_new1 = gpd.GeoDataFrame(row_dict1, crs=edges.crs)
    row_new1.set_index(['u', 'v', 'key'], inplace=True)

    # change the geometry attribute in dictionary with new merged line
    row_dict2['geometry'] = [geom_edge_2]

    # check if length attribute is part of dictionary and update geometry
    if 'length' in row_dict2:
        row_dict2['length'] = round(geom_edge_2.length * (111000), 2) # convert from degrees to meters
    
    row_dict2['u'] = new_edge_2[0]
    row_dict2['v'] = new_edge_2[1]
    row_dict2['key'] = new_edge_2[2]

    # create gdf of new edge and append to edges_gdf
    row_new2 = gpd.GeoDataFrame(row_dict2, crs=edges.crs)
    row_new2.set_index(['u', 'v', 'key'], inplace=True)
    
    if split_type == 'new' or split_type == 'idx':
        return node_new, row_new1, row_new2
    if split_type == 'exist':
        return row_new1, row_new2

def new_edge_straight(new_edge, nodes, edges, group=None):
    """Function creates a new edge given 2 nodes and edge/node gdf

    Args:
        new_edge (tuple): contains the new edge (node1, node2)
        nodes (GeoDataFrame): nodes gdf so that geometry is obtained
        edges (GeoDataFrame): edges gdf to create a dummy dictionary

    Returns:
        GeoDataFrame: A GeoDataFrame of just the new edge. Note that only 
        'geometry', 'length', 'bearing' columns are accurate.
    """    

    # create new geometry for edge
    u_geom = list(nodes.loc[new_edge[0], 'geometry'].coords)
    v_geom = list(nodes.loc[new_edge[1], 'geometry'].coords)
    new_geom = LineString([u_geom[0], v_geom[0]])
 
    # get dummy dictionary to keep consistent columns
    row_dict = edges.iloc[0].to_dict()

    # fill dummy dictionary with correct values
    row_dict['geometry'] = [new_geom]

    # check if length attribute is part of dictionary and update geometry
    if 'length' in row_dict:
        row_dict['length'] = round(new_geom.length * (111000), 2) # convert from degrees to meters
    
    if 'bearing' in row_dict:
        row_dict['bearing'] = round(get_bearing(u_geom[0][::-1], v_geom[0][::-1]), 2)

    if group:
        row_dict['stroke_group'] = group

    row_dict['u'] = new_edge[0]
    row_dict['v'] = new_edge[1]
    row_dict['key'] = 0
    
    row_new = gpd.GeoDataFrame(row_dict, crs=edges.crs)
    row_new.set_index(['u', 'v', 'key'], inplace=True)

    return row_new

def merge_groups(edges, edge_a, edge_b):
    """Code does merging and splitting in a smart way. combines split_group_at_node and merge_to_group
    code is more dynamic. Two types of inputs. If arg_a and arg_b are edges then merge_type should be edges.
    These two edges should be connected at a node but have different group numbers.

    if merge_type = 'strokes' then it uses strokes (TODO) maybe not necessary.

    TODO: make it work with only group inputs

    Args:
        edges (gdf): [description]
        edge_a (tuple/string): tuple of edges or groups to join
        edge_b (tuple/string): tuple of edges or groups to join

    Return
    edge GeoDataFrame
    """    

    # first make copy of edges
    edges_gdf = edges.copy()

    # get node in common between both edges
    node_split = tuple(set(edge_a[:-1]) & set(edge_b[:-1]))[0]

    # get group numbers of both edges
    try:
        group_a = edges_gdf.loc[edge_a].stroke_group
    except KeyError:
        edge_a = (edge_a[1], edge_a[0], 0)
        group_a = edges_gdf.loc[edge_a].stroke_group

    try:
        group_b = edges_gdf.loc[edge_b].stroke_group
    except KeyError:
        edge_b = (edge_b[1], edge_b[0], 0)
        group_b = edges_gdf.loc[edge_b].stroke_group        

    # ensure that group numbers are different
    if group_a == group_b:
        print('Merge groups not needed. Groups are already part of same group')
        return edges_gdf

    # get indices of both groups    
    group_a_uv = list(edges_gdf[edges_gdf['stroke_group']==group_a].index)
    group_b_uv = list(edges_gdf[edges_gdf['stroke_group']==group_b].index)

    # get all nodes (even repeated into a list)
    count_nodes_a = []
    for uv in group_a_uv:
        count_nodes_a.append(uv[0])
        count_nodes_a.append(uv[1])

    count_nodes_b = []
    for uv in group_b_uv:
        count_nodes_b.append(uv[0])
        count_nodes_b.append(uv[1])

    # make countes
    node_count_a = dict(Counter(count_nodes_a))
    node_count_b = dict(Counter(count_nodes_b))

    # if both have a count of 1 they just need to be merged
    if node_count_a[node_split] == 1 and node_count_b[node_split] == 1:

        # groups to_merge
        groups_to_merge = [group_a, group_b]

        # merge_groups
        edges_gdf = merge_to_group(edges_gdf, groups_to_merge)
    
    # only go into loop if any of the node_counts are 1.
    elif node_count_a[node_split] == 1 or node_count_b[node_split] == 1:

        # inital division of group_to_keep and group_to_split. This is just based on which is smaller
        group_to_keep = str(min(int(group_a), int(group_b)))
        group_to_split = str(max(int(group_a), int(group_b)))

        # now check that node_split at group_to_split is not first or last node of group
        split_group_uv = list(edges_gdf[edges_gdf['stroke_group']==group_to_split].index)

        # get all nodes (even repeated into a list)
        count_nodes = []
        for uv in split_group_uv:
            count_nodes.append(uv[0])
            count_nodes.append(uv[1])
        
        # make a counter
        node_count = dict(Counter(count_nodes))

        # now check if the count of node_split is 1
        if node_count[node_split] == 1:
            group_to_split, group_to_keep = group_to_keep, group_to_split

        # split the selected group at the split node.
        edges_gdf = split_group_at_node(edges_gdf, node_split, group_to_split)

        # check if split edge has a new group number
        edge_to_split = edge_a if group_to_split == group_a else edge_b
        potential_new_splt_group = edges_gdf.loc[edge_to_split].stroke_group

        groups_to_merge = [group_to_keep, potential_new_splt_group]

        # merge_groups
        edges_gdf = merge_to_group(edges_gdf, groups_to_merge)
    
    else:

        # both groups should be split
        edges_gdf = split_group_at_node(edges_gdf, node_split, group_a)
        edges_gdf = split_group_at_node(edges_gdf, node_split, group_b)

        # call itself with new group splits (recursive)
        edges_gdf = merge_groups(edges_gdf, edge_a, edge_b)

    return edges_gdf

def merge_to_group(edges, groups_to_merge):
    
    # create copy of edges
    edges_gdf = edges.copy()

    # select the first group of list as the group for all
    groups_to_merge_ints = [int(group_num) for group_num in groups_to_merge]
    merged_group_num = str(min(groups_to_merge_ints))

    # remove merge group number for loop
    groups_to_merge.remove(merged_group_num)

    # loop through all groups except first one
    for group in groups_to_merge:
        group_gdf = edges_gdf[edges_gdf['stroke_group']== group]
        edges_replace = group_gdf.index.values
        
        # change stroke_group to first group
        edges_gdf.loc[edges_replace, 'stroke_group'] = merged_group_num

    return edges_gdf

def split_group_at_node(edges, node_split, curr_group):

    '''Split a group at a node. Create a new group for all edges in front of node. edges must have set direction applied'''

    # create copy of edges
    edges_gdf = edges.copy()

    # get all stroke groups in an array. Then get max value and add 1 for new group number
    all_stroke_groups = np.array(edges_gdf.loc[:,'stroke_group']).astype(np.int32)
    new_group_num = np.amax(all_stroke_groups) + 1

    # find relevant edges of node_split
    edge_uv = list(edges_gdf.index.values)
    edges_with_node = [item for item in edge_uv if node_split in item]

    # remove edges that are not part of group
    edges_group = []

    for edge in edges_with_node:
        group_check = edges_gdf.loc[edge, 'stroke_group']
        if str(group_check) == str(curr_group):
            edges_group.append(edge)

    # get only front edge. It will be the edge that has node to split in first value of edge_a or edge_b
    edge_a = edges_group[0]
    edge_b = edges_group[1]
    edge_front = edge_a if node_split == edge_a[0] else edge_b

    # edge front and subsequent edges get a new group
    current_group = edges_gdf.loc[edge_front, 'stroke_group']        
    group_gdf = edges_gdf[edges_gdf['stroke_group']==current_group]
    group_uv = list(group_gdf.index.values)

    # organize group
    # check if edge in front continues
    end_node = edge_front[1]
    end_edges_to_merge = [edge_front]

    while end_node:
        node_inspect = end_node

        edges_in_merge1 = [item for item in group_uv if node_inspect in item]
        try:
            edge_1 = edges_in_merge1[0]
            edge_2 = edges_in_merge1[1]

            if node_inspect == edge_1[0]:
                edge_front = edge_1
            else:
                edge_front = edge_2
            end_node = edge_front[1]
            end_edges_to_merge.append(edge_front)
        except IndexError:
            end_node = False  


    # see what columns to add in new gdf with dummy_dict
    dummy_edge_dict = edges.loc[edge_front].to_dict()
    column_names = []
    for key in dummy_edge_dict:
        column_names.append(key)


    # edges to alter sub group and initialized geo_data list
    new_group_edges = end_edges_to_merge
    my_geodata = []

    # for loop to create sub geo data list
    for new_edge in new_group_edges:
        sub_geo_data_list = [new_edge[0], new_edge[1], new_edge[2]]
        for attr_key in column_names:
            # change stroke_group in data
            if attr_key == 'stroke_group':
                value = str(new_group_num)
            
            # keep other attributes the same
            else:
                value = group_gdf.loc[new_edge, attr_key]
            sub_geo_data_list.append(value)
        
        # add data to master_geo data
        my_geodata.append(sub_geo_data_list)

    # add u,v, and key to the column nakes
    column_names = ['u', 'v', 'key'] + column_names

    # create edge geodataframe
    new_group_gdf = gpd.GeoDataFrame(my_geodata, columns=column_names, crs=edges.crs)
    new_group_gdf.set_index(['u', 'v', 'key'], inplace=True)

    ### get index values of new group
    new_group_edges = new_group_gdf.index.values

    # drop new_group dataframe and add new_group_gdf to edges_gdf with current stroke_group
    edges_gdf.drop(index=new_group_edges, inplace=True)
    edges_gdf = edges_gdf.append(new_group_gdf)

    return edges_gdf

def get_new_node_osmid(nodes_gdf):
    
    # get max and min node id so we can generate a new node id.
    node_osmids = nodes_gdf.index.values
    max_node_id = np.amax(node_osmids)
    min_node_id = np.amin(node_osmids)

    if min_node_id - 1 > 0: 
        new_node_id = min_node_id - 1
    else:
        new_node_id = max_node_id + 1

    return new_node_id

def get_phantom_intersections(nodes, edges):


    # get node, edge gdf
    nodes_gdf = nodes.copy()
    edges_gdf = edges.copy()
    
    # Run while loop if there are you get some intersections
    while True:

        # get edge indices
        edges_uv = list(edges_gdf.index)

        # get list of geometries and add edge attribute as its id
        geom_list = []
        for index, row in edges_gdf.iterrows():
            
            geom = row.loc['geometry']
            geom.id = index

            geom_list.append(geom)

        # add geometry to an RTREE
        tree = STRtree(geom_list)

        for index_edge, _ in enumerate(edges_uv):
            
            # get info of current edge to check (called main_edge)
            geom_split_main = geom_list[index_edge]
            edge_split_main = geom_split_main.id

            # get first and last point of current linestring to check intersections
            first_point = Point(geom_split_main.coords[0])
            last_point = Point(geom_split_main.coords[-1])

            # get potential interesections in the rtree
            potential_intersections = tree.query(geom_split_main)

            # create empty list to fill with tuple with edge_id and point geometry check actual intersections
            final_intersections = []
            for line_geom in potential_intersections:
                actual_intersections = geom_split_main.intersection(line_geom)

                # check for point intersections
                if actual_intersections.geom_type == 'Point':

                    # check if the intersections are in first or last point
                    if not (actual_intersections.almost_equals(first_point) or actual_intersections.almost_equals(last_point)):
                        final_intersections.append((line_geom.id, actual_intersections, line_geom))

                # check for multipoint intersections
                if actual_intersections.geom_type == 'MultiPoint':
                    # loop through each point and then check that it is not first or last point
                    for sub_point in actual_intersections:
                        if not (sub_point.almost_equals(first_point) or sub_point.almost_equals(last_point)):
                            final_intersections.append((line_geom.id, sub_point, line_geom))

            # only go into loop if there are intersections
            if final_intersections:
                # split edge at first value of final_intersections and then restart loop
                split_info = final_intersections[0]
                edge_split_sec = split_info[0]
                new_point = split_info[1]

                ##################### SPLITTING MAIN EDGE HERE ###############

                # select edge to split, the new geometry and location along linestring
                edge_to_split = edge_split_main
                new_node_osmid = get_new_node_osmid(nodes_gdf)
                split_loc = new_point

                # split edge
                node_new, row_new1, row_new2 = split_edge(edge_to_split, new_node_osmid, nodes_gdf, edges_gdf, split_loc)

                # append nodes and edges and remove split edge
                nodes_gdf = nodes_gdf.append(node_new)
                edges_gdf = edges_gdf.append([row_new1, row_new2])
                edges_gdf.drop(index=edge_to_split, inplace=True)

                ##################### SPLITTING SECONDARY EDGE HERE ###############

                # select edge to split, the new geometry and location along linestring
                edge_to_split = edge_split_sec
                split_loc = new_point
                exist_node_osmid = new_node_osmid

                # # split edge
                row_1, row_2 = split_edge(edge_split_sec, exist_node_osmid, nodes_gdf, edges_gdf, split_loc, split_type='exist')

                # append nodes and edges and remove split edge
                edges_gdf = edges_gdf.append([row_1, row_2])
                edges_gdf.drop(index=edge_split_sec, inplace=True)

                break

        # break while loop if for loop goes through all edges without finding intersection
        if index_edge == len(edges_uv) - 1:
            break

    return nodes_gdf, edges_gdf

def split_line_with_point(line, splitter):
    # code taken from shapely ops.py
    # _split_line_with_point(line, splitter). It did not really work because it get's stuck at first if statement
    # shapely code says point is not directly on line
    # point is on line, get the distance from the first point on line
    distance_on_line = line.project(splitter)
    coords = list(line.coords)
    # split the line at the point and create two new lines
    current_position = 0.0
    for i in range(len(coords)-1):
        point1 = coords[i]
        point2 = coords[i+1]
        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]
        segment_length = (dx ** 2 + dy ** 2) ** 0.5
        current_position += segment_length
        if distance_on_line == current_position:
            # splitter is exactly on a vertex
            return [
                LineString(coords[:i+2]),
                LineString(coords[i+1:])
            ]
        elif distance_on_line < current_position:
            # splitter is between two vertices
            return [
                LineString(coords[:i+1] + [splitter.coords[0]]),
                LineString([splitter.coords[0]] + coords[i+1:])
            ]
    return [line]

def id_parallel_streets(edges, dist_near=32, angle_cut=20):
    
    # get edge indices
    edges_uv = list(edges.index)
    edge_dict = dict()

    # get list of geometries and add edge attribute as its id
    idx_tree = rtree.index.Index()

    # merged gdf
    edges_gdf= gpd.GeoDataFrame(columns=['u', 'v', 'key', 'geometry'], crs=edges.crs)
    edges_gdf.set_index(['u', 'v', 'key'], inplace=True)


    i = 0
    for index, row in edges.iterrows():
        
        geom = row.loc['geometry']
        edge_dict[i] = index
        idx_tree.insert(i, geom.bounds)

        i += 1

    closest_dict = dict()
    for index_edge, row in edges.iterrows():
        geom = row.loc['geometry']
        group1 = row.loc['stroke_group']
        nearest = []
        geom_merge = []
        geom_merge.append(geom)
        lats1, lons1 = geom.xy
        l1 = [[lats1[0], lons1[0]], [lats1[-1], lons1[-1]]]

        j = 1
        while True:

            nearest_trial = list(idx_tree.nearest(geom.bounds, j))
            temp_edge_id = edge_dict[nearest_trial[-1]]
            temp_geom = edges.loc[temp_edge_id, 'geometry']
            group2 = edges.loc[temp_edge_id, 'stroke_group']

            j += 1

            # check intersection
            if temp_geom.intersects(geom):
                continue
            
            # check bearing
            lats2, lons2 = temp_geom.xy
            l2 = [[lats2[0], lons2[0]], [lats2[-1], lons2[-1]]]
            between_angle = btw_angle(l1, l2)
            
            if between_angle > angle_cut:
                continue
            
            if group1 == group2:
                continue
            
            # get nearest points into geometry and get distance in meters
            p1, p2 = ops.nearest_points(geom, temp_geom)
            dist = qdrdist(p1.y, p1.x, p2.y, p2.x)

            if dist <= dist_near:
                nearest.append(temp_edge_id)
                geom_merge.append(temp_geom)
            else:
                break
        
        # only append if nearest exists 
        if nearest:
            closest_dict[index_edge] = nearest
            collection_lines = MultiLineString(geom_merge)

            # create geodatarame
            row_dict = dict()
            row_dict['u'] = index_edge[0]
            row_dict['v'] = index_edge[1]
            row_dict['key'] = index_edge[2]
            row_dict['geometry'] = [collection_lines]

            edge_gdf_new = gpd.GeoDataFrame(row_dict, crs=edges.crs)
            edge_gdf_new.set_index(['u', 'v', 'key'], inplace=True)
            edges_gdf = edges_gdf.append(edge_gdf_new)    
    
    return edges_gdf

def qdrdist(latd1, lond1, latd2, lond2):
    """ Calculate bearing and distance, using WGS'84
        In:
            latd1,lond1 en latd2, lond2 [deg] :positions 1 & 2
        Out:
            qdr [deg] = heading from 1 to 2
            d [nm]    = distance from 1 to 2 in nm """

    # Taken from bluesky

    # res1 for same hemisphere
    res1 = rwgs84(0.5 * (latd1 + latd2))

    # res2 :different hemisphere
    a    = 6378137.0       # [m] Major semi-axis WGS-84
    r1   = rwgs84(latd1)
    r2   = rwgs84(latd2)
    res2 = 0.5 * (abs(latd1) * (r1 + a) + abs(latd2) * (r2 + a)) / \
        (np.maximum(0.000001,abs(latd1) + abs(latd2)))

    # Condition
    sw   = (latd1 * latd2 >= 0.)

    r    = sw * res1 + (1 - sw) * res2

    # Convert to radians
    lat1 = np.radians(latd1)
    lon1 = np.radians(lond1)
    lat2 = np.radians(latd2)
    lon2 = np.radians(lond2)

    # Corrected to avoid "nan" at westward direction
    d = r*np.arccos(np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1) + \
                 np.sin(lat1)*np.sin(lat2))

    return d

def rwgs84(latd):
    """ Calculate the earths radius with WGS'84 geoid definition
        In:  lat [deg] (latitude)
        Out: R   [m]   (earth radius) """

    # Taken from bluesky

    lat    = np.radians(latd)
    a      = 6378137.0       # [m] Major semi-axis WGS-84
    b      = 6356752.314245  # [m] Minor semi-axis WGS-84
    coslat = np.cos(lat)
    sinlat = np.sin(lat)

    an     = a * a * coslat
    bn     = b * b * sinlat
    ad     = a * coslat
    bd     = b * sinlat

    # Calculate radius in meters
    r = np.sqrt((an * an + bn * bn) / (ad * ad + bd * bd))

    return r

def btw_angle(l1,l2):
    # andrei code
    l1 = np.array(l1)
    l2 = np.array(l2)
    m1 = (l1[1,1]-l1[0,1])/(l1[1,0]-l1[0,0])
    m2 = (l2[1,1]-l2[0,1])/(l2[1,0]-l2[0,0])
    angle_rad = abs(np.arctan(m1) - np.arctan(m2))
    angle_deg = angle_rad*180/np.pi
    return angle_deg