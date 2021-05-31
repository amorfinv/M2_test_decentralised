import osmnx as ox
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
from shapely.geometry import shape, LineString
import math
import fiona
import coins

# use osmnx environment here

def main():

    # working path
    working_path = '/Users/andresmorfin/Desktop/M2/test_scenario1/'

    # convert shapefile to shapely polygon
    center_poly = poly_shapefile_to_shapely(f"{working_path}gis/data/street_info/poly_shapefile.shp")

    # create MultiDigraph from polygon
    G = ox.graph_from_polygon(center_poly, network_type='drive', simplify=True)

    # save as osmnx graph
    ox.save_graphml(G, filepath=f'{working_path}gis/data/street_graph/raw_streets.graphml')

    # remove unconnected streets and add edge bearing attrbute 
    G = remove_unconnected_streets(G)
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
    edges = remove_two_way_edges(edges)

    # remove non parallel opposite edges (or long way)
    edges = remove_long_way_edges(edges)

    # allocated edge height based on cardinal method
    layer_allocation, _ = allocate_edge_height(edges, 0)

    # Assign layer allocation to geodataframe
    edges['layer_height'] = layer_allocation

    # # add interior angles at all intersections
    edges = add_edge_interior_angles(edges)

    # Perform COINS algorithm to add stroke groups
    coins.COINS(edges)

    # create graph and save edited
    G = ox.graph_from_gdfs(nodes, edges)
    # save as osmnx graph
    ox.save_graphml(G, filepath=f'{working_path}gis/data/street_graph/processed_graph.graphml')

    # Save geopackage for import to QGIS and momepy
    ox.save_graph_geopackage(G, filepath=f'{working_path}gis/data/street_graph/processed_graph.gpkg')

    # save projected
    G_projected = ox.project_graph(G)
    ox.save_graph_geopackage(G_projected, filepath=f'{working_path}gis/data/street_graph/projected_graph.gpkg')

    # save csv for reference
    edges.to_csv(f'{working_path}gis/data/street_graph/edges.csv')
    nodes.to_csv(f'{working_path}gis/data/street_graph/nodes.csv')


# general TODO: perhas no need to convert to geodataframe

def poly_shapefile_to_shapely(filepath):
    '''
    Convert a polygon shapefile into a shapely polygon. This is done so that a QGIS shapefile can be used to generate a street network
    Usage:
        shapely_polygon = poly_shapefile_to_shapely(shapefile_polygon_filename)
    '''

    # read shapefile and make into a dictionary
    shapefile = fiona.open(filepath)
    shap_dict = next(iter(shapefile))

    # get geometry coordinates and convert to shapely
    poly = shape(shap_dict['geometry'])

    return poly

def remove_unconnected_streets(G):
    '''
    Argument is an osmnx Graph.

    Removes unconnected streets (dead-ends) from graph. 
    It searches for street_count = 1 inside the node geodataframe and removes them from graph.

    Returns an osmnx Graph

    TODO: remove left-over streets that are one node once nodes are initially removed.
    So if node is in a two way street that needs to be removed it should only appear twice in the edges dataframe.
    Condition for dead ends that are two way streets. Remove node ids that appear on edges twice. 
    However, must be careful if doing it for a subset graph because it may cut off nodes that are not dead-ends 
    but have an edge that connect to an node outside subset graph. Perhaps the way to do this:
        1) Select nodes from subset graph that appear once but have a street count larger than 1.
           This basically means that they appear as one way streets in the graph but are really not.
           Ensure that these nodes should not be deleted.
        2) Select nodes with street count of 1 and remove them from graph.
        3) once these nodes are removed check, time to check if there are any new dead-ends as a result of the first deletion.
        4) Condition for dead ends with two way streets. Remove nodes that appear on edges twice. Make sure they are not
           in node list from step 1.
        5) Condition for dead ends with one way streets. Remove nodes that appear on edges once. Make sure they are not in
           node list from step 1.
    Possible issues. step 3-5 are an iteration. Because deletions may trigger new dead ends. Possible issue is that this loop continues
    until all edges are removed. So it is important to keep a count of edges and nodes removed to do a check at the end and see 
    how many were removed. Ideally it is just a small percentage.
    If it is a large percentage the unnconencted street removal is not useful and another method should be found.
    Perhaps a more manual way with QGIS
    
    '''

    # convert to geodataframe for easier manipulation
    g = ox.graph_to_gdfs(G)

    # Get node data from geodataframe
    nodes = g[0]
    edges = g[1]

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

    # get bearing difference index
    bearing_indices = calculate_bearing_diff_indices(bearings, low_bound_1, high_bound_1, low_bound_2, high_bound_2)
    
    return layer_allocation, bearing_indices

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
    # perhaps try to use previous calculations to slow calculation down
    # add column to dataframes with internal angle between edges
    # try edge with just two points. Should work
    
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
    
    # add interior angle dictionary to edges
    edges['edge_interior_angle'] = edge_interior_angle

    return edges

"""
The below function calculates the joining angle between
two line segments. FROM COINS
"""
def angleBetweenTwoLines(line1, line2):
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

"""
This below function calculates the acute joining angle between
two given set of points. FROM COINS
"""
def pointsSetAngle(line1, line2):
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

# FROM COINS

def computeOrientation(line):
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

"""
The function below calculates the angle between two points in space. FROM COINS
"""

def computeAngle(point1, point2):
    height = abs(point2[1] - point1[1])
    base = abs(point2[0] - point1[0])
    angle = round(math.degrees(math.atan(height/base)), 3)
    return(angle)

def remove_two_way_edges(edges):

    # create sub geodataframe with only oneway== False streets
    one_way_bool = edges[edges["oneway"] == False]["oneway"]

    # isolate u,v components
    edge_indices = list(one_way_bool.index.values)

    # initiaite list that will contain edges to be removed
    edges_remove = []

    # while loop it will check all edges that are duplicate and remove
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
    
    # remove from geodataframe
    edges.drop(edges_remove, inplace=True)

    # set one-way column to true. unsure if this step is necessary
    edges["oneway"].values[:] =  True

    return edges

def remove_long_way_edges(edges):

    '''
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
    '''

    # create sub geodataframe with only oneway== True streets
    one_way_bool = edges[edges["oneway"] == True]["oneway"]

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
            length_to_check = edges.loc[edge_to_check, 'length']
            length_sel = edges.loc[edge_selected[0], 'length']
            
            # remove edge with longer length and add to list
            edge_to_remove = edge_to_check if length_to_check > length_sel else edge_selected[0]
            edges_remove.append(edge_to_remove)
        
        # Delete the edge to check from list before restarting loop
        edge_indices.remove(edge_to_check)

    # remove from geodataframe
    edges.drop(edges_remove, inplace=True)

    return edges

if __name__ == '__main__':
    main()