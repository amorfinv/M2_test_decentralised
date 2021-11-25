import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
import graph_funcs
from os import path
from shapely.geometry import LineString, MultiLineString

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
    G = ox.load_graphml(filepath=path.join(gis_data_path, 'layer_heights.graphml'))

    # convert to gdgs
    nodes, edges = ox.graph_to_gdfs(G)

    # simplify street linestring geometry. Select 1.5 degrees
    angle_threshold = 1.5
    edges_simplified = simplify_street_geometry(edges, 'angle', angle_threshold)

    # convert back to graph and save
    G = ox.graph_from_gdfs(nodes, edges_simplified)

    ox.save_graphml(G, filepath=path.join(gis_data_path, 'layer_heights_simplified.graphml'))

    ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'layer_heights_simplified.gpkg'), directed=True)

def simplify_street_geometry(edges, sim_type = 'shapely', angle_threshold = 1):
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
    angle_threshold : float
        Angle threshold.
    
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
        edges_simplified['geometry'] = edges_simplified.apply(lambda x: simplify_geometry(x['geometry'], angle_threshold), axis=1)

    return edges_simplified

def simplify_geometry(line_geometry, angle_threshold):
    """
    Simplify geometry with angle

    Parameters
    ----------
    line_geometry : LineString
        Line geometry.
    angle_threshold : float
        Angle threshold.

    Returns
    """

    points_to_remove = True

    while points_to_remove:

        # split shapely linestring into segments
        list_points = list(line_geometry.coords)

        # calculate the angle between all line segments
        # get angles from list of points
        angles = angles_from_list_points(list_points)
        
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


def angles_from_list_points(list_points):
    """
    Calculate angles from list of points.

    Parameters
    ----------
    list_points : list
        List of points.

    Returns
    -------
    angles : list
        List of angles.
    """
    angles = []
    for i in range(len(list_points)-2):
        line_segment_1 = LineString([list_points[i], list_points[i+1]]).coords
        line_segment_2 = LineString([list_points[i+1], list_points[i+2]]).coords

        angles.append(180 - graph_funcs.angleBetweenTwoLines(line_segment_1, line_segment_2))
    return angles



if __name__ == '__main__':
    main()