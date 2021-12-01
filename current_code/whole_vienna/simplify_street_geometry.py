import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString, Polygon, LinearRing
import graph_funcs
from os import path
import numpy as np
from momepy import COINS

"""
Code takes geometry of street and returns a simplified version of it.
It looks at the bearing difference between two line segments and
if the difference is less than the threshold then it removes that point
from the line.

THIS IS THE FINAL GRAPH
"""
def main():
    # working path
    gis_data_path = 'gis'

    # Load MultiDigraph from create_graph.py
    G = ox.load_graphml(filepath=path.join(gis_data_path, 'layer_heights_connected.graphml'))

    # convert to gdgs
    nodes, edges = ox.graph_to_gdfs(G)

    # # run coins
    # print('Number of waypoints in original geometry:')
    # coins_obj = COINS(edges)
    # coins_obj.stroke_gdf()
    # print('------------------------------------')

    # simplify street linestring geometry. Select 1.5 degrees
    angle_threshold = 1.5
    edges_simplified = simplify_street_geometry(edges, 'angle', angle_threshold)

    # # run coins
    # print('Number of waypoints in angle simplified geometry:')
    # coins_obj = COINS(edges_simplified)
    # coins_obj.stroke_gdf()
    # print('------------------------------------')

    # first convert to a projected crs
    edges_simplified = edges_simplified.to_crs(epsg=32633)

    # now simplify street geometry with triangles
    distance_threshold = 2
    edges_simplified = simplify_street_geometry(edges_simplified, 'triangles', distance_threshold)

    # convert back to epsg 4326
    edges_simplified = edges_simplified.to_crs(epsg=4326)
   
    # print('Number of waypoints in distance simplified geometry:')
    # coins_obj = COINS(edges_simplified)
    # coins_obj.stroke_gdf()

    # convert back to graph and save
    G = ox.graph_from_gdfs(nodes, edges_simplified)

    ox.save_graphml(G, filepath=path.join(gis_data_path, 'layer_heights_simplified_2.graphml'))

    ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'layer_heights_simplified_2.gpkg'), directed=True)

def simplify_street_geometry(edges, sim_type = 'shapely', threshold = 1):
    """
    Simplify street geometry.

    Parameters
    ----------
    edges : GeoDataFrame
        Edges of the graph.
    sim_type : str
        Type of simplification.
        'shapely' - use shapely simplification.
        'angle' - use angle simplification.
        'triangles' - use triangles simplification.
    threshold : float
        Angle or distance threshold dependin on sim_type.
    
    Returns
    -------
    edges_simplified : GeoDataFrame
        Simplified edges.
    """
    edges_gdf = edges.copy()

    if sim_type == 'shapely':

        # simplify street geometry
        # convert to crs 32633
        edges_gdf = edges_gdf.to_crs(epsg=32633)

        edges_simplified = edges_gdf.copy()
        edges_simplified['geometry'] = edges_simplified.apply(lambda x: x['geometry'].simplify(tolerance=4), axis=1)

        edges_gdf = edges_gdf.to_crs(epsg=4326)


    elif sim_type == 'angle':

        edges_simplified = edges_gdf.copy()
        edges_simplified['geometry'] = edges_simplified.apply(lambda x: simplify_geometry_angle(x['geometry'], threshold), axis=1)

    elif sim_type == 'triangles':
        # this goes into all triangles
        
        edges_simplified = edges_gdf.copy()
        edges_simplified['geometry'] = edges_simplified.apply(lambda x: simplify_geometry_triangle(x['geometry'], threshold), axis=1)

    return edges_simplified


def simplify_geometry_triangle(line_geometry, distance_threshold):
    """
    Simplify geometry with angle

    Parameters
    ----------
    line_geometry : LineString
        Line geometry.
    distance_threshold : float
        Distance threshold.

    Returns
    """

    # split shapely linestring into segments
    list_points = list(line_geometry.coords)

    # go into while loop only if there are more than 2 points on linestring
    if len(list_points) > 3:
        points_to_remove = True
    else:
        points_to_remove = False

    # start with first angle
    angle_idx = 0
    while points_to_remove:

        # split shapely linestring into segments
        list_points = list(line_geometry.coords)

        # calculate the angle between all line segments
        angles = angles_from_list_points(list_points, angle_choice='int_angle')

        for angle_idx, angle in enumerate(angles):

            # angle number inside linestring add 1 so it corresponds to the index of the point
            angle_num = angle_idx + 1

            # create a linearring
            triangle_ring = LinearRing([list_points[angle_num-1], 
                                        list_points[angle_num], 
                                        list_points[angle_num+1],
                                        list_points[angle_num-1]])
            
            triangle_coords = list(triangle_ring.coords)
            # the location of the angle in the linear ring
            ref_angle_loc = 1

            # get hypotenuse
            hypotenuse = LineString([triangle_coords[0] , triangle_coords[1]]).length

            # make linear ring counter clockwise if it is not
            if triangle_ring.is_ccw:
                triangle_ring.coords = list(triangle_ring.coords)[::-1]

                # move location of angle in linear ring
                ref_angle_loc = 2

                # get hypotenuse
                hypotenuse = LineString([triangle_coords[0] , triangle_coords[2]]).length
            
            # get interior angles of triangle
            interior_angles = interior_angles_from_triangle(triangle_ring)

            # get distance from potential point to remove to potential new linestring
            dist_to_remove = hypotenuse*np.sin(np.deg2rad(interior_angles[0]))
            
            # if removing the point, break the for loop to start over with new geometry
            if dist_to_remove < distance_threshold:

                # remove from list of points
                list_points.pop(angle_num)

                # create new linestring with less points
                line_geometry = LineString(list_points)

                break 
                
            else:
                # continue the for loop to check next angles
                pass

        # if have gone through all angles then break the loop
        if angle_idx == len(angles)-1:
            points_to_remove = False

    return line_geometry

def interior_angles_from_triangle(triangle_ring):
    """
    Get interior angles of triangle.

    Parameters
    ----------
    triangle_ring : LinearRing
        Triangle ring.

    Returns
    -------
    interior_angles : list
        Interior angles of triangle.
    """
    # get points of triangle ring
    points = list(triangle_ring.coords)

    # get points of triangle
    triangle_point_0 = [points[2], points[0], points[1]]
    triangle_point_1 = [points[0], points[1], points[2]]
    triangle_point_2 = [points[1], points[2], points[0]]

    triangle_points = [triangle_point_0, triangle_point_1, triangle_point_2]
    
    # get interior angles of triangle
    interior_angles = []

    for point_list in triangle_points:

        # get ordered points
        p1 = point_list[0]
        ref = point_list[1]
        p2 = point_list[2]

        x1, y1 = p1[0] - ref[0], p1[1] - ref[1]
        x2, y2 = p2[0] - ref[0], p2[1] - ref[1]
        
        # calculate the angle between the two vectors with dot product
        numer = (x1 * x2 + y1 * y2)
        denom = np.sqrt((x1 ** 2 + y1 ** 2) * (x2 ** 2 + y2 ** 2))
        angle = np.arccos(numer / denom) 

        # convert angle to degrees
        angle = round(np.rad2deg(angle),3)

        # add angle to list
        interior_angles.append(angle)

    return interior_angles


def simplify_geometry_angle(line_geometry, angle_threshold):
    """
    Simplify geometry with angle

    Parameters
    ----------
    line_geometry : LineString
        Line geometry.
    angle_threshold : float
        Angle threshold.

    Returns
    -------
    line_geometry : LineString
    """

    points_to_remove = True

    while points_to_remove:

        # split shapely linestring into segments
        list_points = list(line_geometry.coords)

        # calculate the angle between all line segments
        # get angles from list of points
        angles = angles_from_list_points(list_points, angle_choice='angle_diff')
        
        # remove points with angle < angle_threshold
        points_to_remove = []
        for angle_idx, angle in enumerate(angles):

            # angle number inside linestring
            angle_num = angle_idx + 1
            
            # if the angle is less than the thresold, remove the point from list_points
            if angle < angle_threshold:

                points_to_remove.append(angle_num)

        # sort points to remove backwards
        if len(points_to_remove) > 0:
            points_to_remove = sorted(points_to_remove,reverse=True)
        else:
            # break while loop if there are no more points to remove
            break
        
        # remove useless points from points_to_remove
        for point_num in points_to_remove:
            # remove point from list
            list_points.pop(point_num)

        # create new linestring with less points
        line_geometry = LineString(list_points)

    return line_geometry

def angles_from_list_points(list_points, angle_choice='angle_diff'):
    """
    Calculate angles from list of points.

    Parameters
    ----------
    list_points : list
        List of points.
    angle_choice : str
        Choice of angle calculation.

    Returns
    -------
    angles : list
        List of angles.
    """

    angles = []
    for i in range(len(list_points)-2):
        line_segment_1 = LineString([list_points[i], list_points[i+1]]).coords
        line_segment_2 = LineString([list_points[i+1], list_points[i+2]]).coords

        if angle_choice == 'angle_diff':
            angles.append(180 - graph_funcs.angleBetweenTwoLines(line_segment_1, line_segment_2))
        elif angle_choice == 'int_angle':
            angles.append(graph_funcs.angleBetweenTwoLines(line_segment_1, line_segment_2))

    return angles


if __name__ == '__main__':
    main()