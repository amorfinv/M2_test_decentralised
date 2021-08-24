from re import split
import osmnx as ox
import geopandas as gpd
import graph_funcs
from os import path, truncate
import math
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely import ops
import pandas as pd
import collections

'''
Initial processing for graph of Vienna:

Steps:
    1) Download from OSM or load graph from local.
    2) Save graphml and geopackage as "raw_streets*"
    3) Run graph_funcs.initial_prep() to remove some useless attributes.
    4) Run graph_funcs.remove_two_way_edges() and graph_funcs.remove_long_way_edges() to 
       convert graph from MultiDiGraph to DiGraph. Note that in name it is still a MultiDigraph
       but it is actually a directed graph.
    5) Save graphml and geopackage as "initial_processing*".
'''
def main():

    # working path
    gis_data_path = 'gis'

    # get airspace polygon from geopackage
    airspace_gdf = gpd.read_file(path.join(gis_data_path, 'airspace', 'small_area.gpkg'))
    airspace_gdf.to_crs("EPSG:4326", inplace = True)
    
    airspace_poly = airspace_gdf.geometry.iloc[0]

    # create MultiDigraph from polygon
    raw_graph_path = path.join(gis_data_path, 'streets', 'small_test.graphml')

    # check if graph alrrady downloaded
    if path.exists(raw_graph_path):
        G = ox.load_graphml(raw_graph_path)
    else:
        print('DOWNLOADING!')
        G = ox.graph_from_polygon(airspace_poly, network_type='drive', simplify=True, clean_periphery=True, truncate_by_edge=True)
    
        # save as osmnx graph
        ox.save_graphml(G, filepath=raw_graph_path)
        ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'streets', 'small_test.gpkg'), directed=True)
    
    # first order of buisness after import of graph is to remove unused attributes from graph
    G = graph_funcs.initial_prep(G)

    # convert graph to node, edge geodataframe
    nodes, edges = ox.graph_to_gdfs(G)
    
    # remove two way edges
    edges = graph_funcs.remove_two_way_edges(edges)
    
    # remove non parallel opposite edges (or long way)
    edges = graph_funcs.remove_long_way_edges(edges)

    # add new nodes at phantom intersections
    nodes, edges = get_phantom_intersections(nodes, edges)

    # create graph and save edit
    G = ox.graph_from_gdfs(nodes, edges)
    
    # save as osmnx graph
    ox.save_graphml(G, filepath=path.join(gis_data_path, 'streets', 'small_test2.graphml'))
    
    # Save geopackage for import to QGIS
    ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'streets', 'small_test2.gpkg'), directed=True)

def get_phantom_intersections(nodes, edges):

    from shapely.strtree import STRtree
    from shapely.geometry import Point 

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

def split_edge(edge_to_split, node_osmid, nodes, edges, split_loc=None, split_type='new'):
    '''

    function splits edge and returns new node geometry and new edges,
    at the moment delete edges outside of this function. At the moment it only works with linestrings
    cannot split a straight line yet

    split_loc is the index of a point in the linestring if split_type = 'idx'
    split_loc is a new Point if split_type='new'
 

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

if __name__ == '__main__':
    main()

