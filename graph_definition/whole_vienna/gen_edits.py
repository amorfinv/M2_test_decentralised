import os
from platform import node
from networkx.classes.function import degree
import osmnx as ox
import geopandas as gpd
import networkx as nx
from osmnx.projection import project_gdf
from osmnx.utils_graph import graph_from_gdfs
from shapely.geometry.point import Point
import graph_funcs
from os import path
import math
import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely import ops
import pandas as pd
import collections
# use osmnx environment here

'''
Prepare graph for genetic algorithm.

Steps:
    1) Load 'cleaning_process_2.graphml'.
'''

def main():

    # working path
    gis_data_path = 'gis'

    # get airspace polygon from geopackage
    edges = gpd.read_file(path.join(gis_data_path, 'streets', 'gen_edits.gpkg'), layer='edges')
    nodes = gpd.read_file(path.join(gis_data_path, 'streets', 'gen_edits.gpkg'), layer='nodes')
    
    # recreate edges in format for osmnx
    edges = graph_funcs.edge_gdf_format_from_gpkg(edges)
    nodes = graph_funcs.node_gdf_format_from_gpkg(nodes)

    ##### edit 1
    edge_a = (60216504, 60216511, 0)
    edge_b = (60216542, 60216511, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)

    edge_direction = (60216507, 2600415942, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)
    
    #### edit 2
    edge_direction = (2397500105, 2397733665, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)
    
    #### edit 3
    edge_a = (2605251576, 2424032524, 0)
    edge_b = (2424032524, 33079197, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)
    
    
    #### edit 4
    edge_direction = (25280889, 398002, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)
    
    #### edit 5
    edge_direction = (27375757, 93279698, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)
    
    #### edit 6
    edge_direction = ( 30685749, 30685743, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)
    
    #### edit 7
    edge_direction = ( 1476901999, 395907, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)
    
    #### edit 8
    edge_a = (813850447, 199678, 0)
    edge_b = (27377268, 813850447, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)
    
    #### edit 9
    edge_a = (48753805, 48753809, 0)
    edge_b = (48753805, 48753788, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)
    
    edge_direction = (48753809, 48753805, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)
    
    #### edit 10
    edge_direction = (123430807, 123610502, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)
    
    #### edit 11
    edge_a = (2389850711, 35675056, 0)
    edge_b = (2389850711, 2389850721, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)
    
    edge_direction = (59988314, 2389850716, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)
    
    #### edit 12
    edge_direction = (353280664, 60637199, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)
    
    #### edit 13
    edge_a = (252621941, 546942312, 0)
    edge_b = (252621941, 252622009, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)
    
    edge_direction = (254314477, 60637196, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)
    
    #### edit 14
    edge_direction = (684998, 685001, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)
    
    #### edit 15
    edge_a = (60211202, 2388913465, 0)
    edge_b = (2388913465, 8173811392, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)
    
    #### edit 16
    edge_a = (59987467, 60210198, 0)
    edge_b = (60210198, 1912541339, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)
    
    #### edit 17
    edge_a = (8296326836, 8300082985, 0)
    edge_b = (8300082985, 4089671728, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)
    
    #### edit 18
    edge_a = (8406077356, 60210120, 0)
    edge_b = (60210120, 8296326836, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)
    
    #### edit 19 (big one)
    edge_a = (2388183609, 60733317, 0)
    edge_b = (33254354, 60733317, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)
    
    edge_c = (60627223, 3672898099, 0)
    edge_d = (3672898099, 33370280, 0)
    edges = graph_funcs.merge_groups(edges, edge_c, edge_d)
    
    edge_direction = (199727, 33472242, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)
    
    #### edit 20
    edge_direction = (513404700, 684994, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)
    
    #### edit 21
    edge_a = (1188875861, 2389831852, 0)
    edge_b = (1188875861, 330372031, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)
    
    edge_direction = (321552008, 321552200, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)
    
    #### edit 22
    edge_direction = (33196469, 33196457, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)
    
    #### edit 23
    edge_direction = (259192914, 199732, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)
    
    #### edit 24
    edge_a = (9921163, 16054998, 0)
    edge_b = (623411216, 16054998, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)
    
    edge_direction = (24967092, 17312867, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)
    
    #### edit 25
    edge_a = (9921091, 685238, 0)
    edge_b = (685238, 685244, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)
    
    #### edit 25
    edge_a = (392482, 65011391, 0)
    edge_b = (65011391, 4932689109, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)
    
    #### edit 26
    edge_direction = (333305349, 26405234, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)
    
    #### edit 27
    edge_direction = (199684, 24950487, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)
    
    #### edit 28
    node_a = 2554619250
    node_b = 2554619257
    edges = graph_funcs.connect_nodes(nodes, edges, [(node_a, node_b)], 1224)
    
    #### edit 29
    edge_a = (1852755631, 123277760, 0)
    edge_b = (123277760, 2106209116, 0)
    edges = graph_funcs.merge_groups(edges, edge_a, edge_b)
    
    #### edit 30
    node_split = 8790237562
    edges = graph_funcs.split_group_at_node(edges, node_split, 103)
    
    edge_direction = (199691, 199678, 0)
    edges = graph_funcs.set_direction_group(nodes, edges, edge_direction)
    

    # remove degree 2 edges with int angle greater than 120 degrees
    # nodes, edges = graph_funcs.simplify_graph(nodes, edges, angle_cut_off=120)

    # create graph and save edited
    G = ox.graph_from_gdfs(nodes, edges)

    # save as osmnx graph
    ox.save_graphml(G, filepath=path.join(gis_data_path, 'streets', 'regrouping_3.graphml'))

    # Save geopackage for import to QGIS a
    ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'streets', 'regrouping_3.gpkg'), directed=True)

    
class COINS:

    """
    Calculates natural continuity and hierarchy of street networks in given
    GeoDataFrame using COINS algorithm.
    For details on the algorithms refer to the original paper :cite:`tripathy2020open`.
    This is a reimplementation of the original script from
    https://github.com/PratyushTripathy/COINS
    ``COINS`` can return final stroke geometry (``.stroke_gdf()``) or a pandas
    Series encoding stroke groups onto the original input geometry
    (``.stroke_attribute()``).
    Parameters
    ----------
    edge_gdf : GeoDataFrame
        GeoDataFrame containing edge geometry of street network. edge_gdf should
        ideally not contain MultiLineStrings.
    angle_threshold : int, float (default 0)
        the angle threshold for the COINS algorithm.
        Segments will only be considered a part of the same street if the
        deflection angle is above the threshold.
    Examples
    --------
    Initialise COINS class. This step will already compute the topology.
    >>> coins = momepy.COINS(streets)
    To get final stroke geometry:
    >>> stroke_gdf = coins.stroke_gdf()
    To get a Series encoding stroke groups:
    >>> stroke_attr = coins.stroke_attribute()
    """

    def __init__(self, edge_gdf, angle_threshold=0):
        self.edge_gdf = edge_gdf
        self.gdf_projection = self.edge_gdf.crs
        self.already_merged = False

        # get indices of original gdf
        self.uv_index = self.edge_gdf.index.tolist()

        # get line segments from edge gdf
        self.lines = [list(value[1].coords) for value in edge_gdf.geometry.iteritems()]

        # split edges into line segments
        self._split_lines()

        # # create unique_id for each individual line segment
        self._unique_id()

        # # compute edge connectivity table
        self._get_links()

        # # find best link at every point for both lines
        self._best_link()

        # # cross check best links and enter angle threshold for connectivity
        self._cross_check_links(angle_threshold)

    def _premerge(self):
        """
        Returns a GeoDataFrame containing the individual segments with all underlying
        information. The result is useful for debugging purposes.
        """
        return self._create_gdf_premerge()

    def stroke_gdf(self):
        """
        Returns a GeoDataFrame containing merged final stroke geometry.
        """
        if not self.already_merged:
            self._merge_lines()
        return self._create_gdf_strokes()

    def stroke_attribute(self):
        """
        Returns a pandas Series encoding stroke groups onto the original input geometry.
        """
        if not self.already_merged:
            self._merge_lines()
        return self._add_gdf_stroke_attributes()

    def _split_lines(self):
        out_line = []
        self.temp_array = []
        n = 0
        # Iterate through the lines and split the edges
        idx = 0
        for line in self.lines:
            for part in _list_to_pairs(line):
                out_line.append(
                    [
                        part,
                        _compute_orientation(part),
                        list(),
                        list(),
                        list(),
                        list(),
                        list(),
                        list(),
                        self.uv_index[idx],
                    ]
                )
                # merge the coordinates as string, this will help in finding adjacent
                # edges in the function below
                self.temp_array.append(
                    [n, f"{part[0][0]}_{part[0][1]}", f"{part[1][0]}_{part[1][1]}"]
                )
                n += 1
            idx += 1

        self.split = out_line

    def _unique_id(self):
        # Loop through split lines, assign unique ID and
        # store inside a list along with the connectivity dictionary
        self.unique = dict(enumerate(self.split))

    def _get_links(self):
        self.temp_array = np.array(self.temp_array, dtype=object)

        items = collections.defaultdict(set)
        for i, vertex in enumerate(self.temp_array[:, 1]):
            items[vertex].add(i)
        for i, vertex in enumerate(self.temp_array[:, 2]):
            items[vertex].add(i)

        p1 = []
        for i, vertex in enumerate(self.temp_array[:, 1]):
            item = list(items[vertex])

            item.remove(i)
            p1.append(item)

        p2 = []
        for i, vertex in enumerate(self.temp_array[:, 2]):
            item = list(items[vertex])

            item.remove(i)

            p2.append(item)

        self.result = list(zip(range(len(p1)), p1, p2))

        for a in self.result:
            n = a[0]
            self.unique[n][2] = a[1]
            self.unique[n][3] = a[2]

    def _best_link(self):
        self.angle_pairs = dict()
        for edge in range(0, len(self.unique)):
            p1_angle_set = []
            p2_angle_set = []

            """
            Instead of computing the angle between the two segments twice, the method
            calculates it once and stores in the dictionary for both the keys. So that
            it does not calculate the second time because the key is already present in
            the dictionary.
            """
            for link1 in self.unique[edge][2]:
                self.angle_pairs["%d_%d" % (edge, link1)] = _angle_between_two_lines(
                    self.unique[edge][0], self.unique[link1][0]
                )
                p1_angle_set.append(self.angle_pairs["%d_%d" % (edge, link1)])

            for link2 in self.unique[edge][3]:
                self.angle_pairs["%d_%d" % (edge, link2)] = _angle_between_two_lines(
                    self.unique[edge][0], self.unique[link2][0]
                )
                p2_angle_set.append(self.angle_pairs["%d_%d" % (edge, link2)])

            """
            Among the adjacent segments deflection angle values, check for the maximum
            value at both the ends. The segment with the maximum angle is stored in the
            attributes to be cross-checked later for before finalising the segments at
            both the ends.
            """
            if len(p1_angle_set) != 0:
                val1, idx1 = max((val, idx) for (idx, val) in enumerate(p1_angle_set))
                self.unique[edge][4] = self.unique[edge][2][idx1], val1
            else:
                self.unique[edge][4] = "dead_end"

            if len(p2_angle_set) != 0:
                val2, idx2 = max((val, idx) for (idx, val) in enumerate(p2_angle_set))
                self.unique[edge][5] = self.unique[edge][3][idx2], val2
            else:
                self.unique[edge][5] = "dead_end"

    def _cross_check_links(self, angle_threshold):
        for edge in range(0, len(self.unique)):

            best_p1 = self.unique[edge][4][0]
            best_p2 = self.unique[edge][5][0]

            if (
                isinstance(best_p1, int)
                and edge in [self.unique[best_p1][4][0], self.unique[best_p1][5][0]]
                and self.angle_pairs["%d_%d" % (edge, best_p1)] > angle_threshold
            ):
                self.unique[edge][6] = best_p1
            else:
                self.unique[edge][6] = "line_break"

            if (
                isinstance(best_p2, int)
                and edge in [self.unique[best_p2][4][0], self.unique[best_p2][5][0]]
                and self.angle_pairs["%d_%d" % (edge, best_p2)] > angle_threshold
            ):
                self.unique[edge][7] = best_p2
            else:
                self.unique[edge][7] = "line_break"

    def _merge_lines(self):
        self.merging_list = list()
        self.merged = list()
        self.edge_idx = list()

        self.result = [
            _merge_lines_loop(n, self.unique) for n in range(len(self.unique))
        ]

        for temp_list in self.result:
            if temp_list not in self.merging_list:
                self.merging_list.append(temp_list)
                self.merged.append(
                    {_list_to_tuple(self.unique[key][0]) for key in temp_list}
                )

                # assign stroke number to edge from argument
                self.edge_idx.append({self.unique[key][8] for key in temp_list})

        self.merged = dict(enumerate(self.merged))
        self.edge_idx = dict(enumerate(self.edge_idx))
        self.already_merged = True

    # Export geodataframes, 3 options
    def _create_gdf_premerge(self):
        # create empty list to fill out
        my_list = []

        for parts in range(0, len(self.unique)):
            # get all segment points and make line
            line_list = _tuple_to_list(self.unique[parts][0])
            geom_line = LineString([(line_list[0]), (line_list[1])])

            # get other values for premerged
            _unique_id = parts
            orientation = self.unique[parts][1]
            links_p1 = self.unique[parts][2]
            links_p2 = self.unique[parts][3]
            best_p1 = self.unique[parts][4]
            best_p2 = self.unique[parts][5]
            p1_final = self.unique[parts][6]
            p2_final = self.unique[parts][7]

            # append list
            my_list.append(
                [
                    _unique_id,
                    orientation,
                    links_p1,
                    links_p2,
                    best_p1,
                    best_p2,
                    p1_final,
                    p2_final,
                    geom_line,
                ]
            )

        edge_gdf = gpd.GeoDataFrame(
            my_list,
            columns=[
                "_unique_id",
                "orientation",
                "links_p1",
                "links_p2",
                "best_p1",
                "best_p2",
                "p1_final",
                "p2_final",
                "geometry",
            ],
            crs=self.gdf_projection,
        )
        edge_gdf.set_index("_unique_id", inplace=True)

        return edge_gdf

    def _create_gdf_strokes(self):

        # create empty list to fill
        my_list = []

        # loop through merged geometry
        for a in self.merged:

            # get all segment points and make line strings
            linelist = _tuple_to_list(list(self.merged[a]))
            list_lines_segments = []

            for b in linelist:
                list_lines_segments.append(LineString(b))

            # merge seperate segments
            geom_multi_line = ops.linemerge(MultiLineString(list_lines_segments))

            # get other values for gdf
            id_value = a
            n_segments = len(self.merged[a])

            # append list
            my_list.append([id_value, n_segments, geom_multi_line])

        edge_gdf = gpd.GeoDataFrame(
            my_list,
            columns=["stroke_group", "n_segments", "geometry"],
            crs=self.gdf_projection,
        )
        edge_gdf.set_index("stroke_group", inplace=True)

        return edge_gdf

    def _add_gdf_stroke_attributes(self):

        # Invert self.edge_idx to get a dictionary where the key is the original edge
        # index and the value is the group
        inv_edges = {
            value: key for key in self.edge_idx for value in self.edge_idx[key]
        }

        # create empty list that will contain attributes of stroke
        stroke_group = list()

        for edge in self.uv_index:
            stroke_group.append(inv_edges[edge])

        return pd.Series(stroke_group, index=self.edge_gdf.index)


"""
The imported shapefile lines comes as tuple, whereas
the export requires list, this finction converts tuple
inside lines to list
"""


def _tuple_to_list(line):
    for a in range(0, len(line)):
        line[a] = list(line[a])
    return line


def _list_to_tuple(line):
    for a in range(0, len(line)):
        line[a] = tuple(line[a])
    return tuple(line)


"""
The below function takes a line as an input and splits
it at every point.
"""


def _list_to_pairs(in_list):
    out_list = []
    index = 0
    for index in range(0, len(in_list) - 1):
        temp_list = [list(in_list[index]), list(in_list[index + 1])]
        out_list.append(temp_list)
    return out_list


"""
The function below calculates the angle between two points in space.
"""


def _compute_angle(point1, point2):
    height = abs(point2[1] - point1[1])
    base = abs(point2[0] - point1[0])
    angle = round(math.degrees(math.atan(height / base)), 3)
    return angle


"""
This function calculates the orientation of a line segment.
Point1 is the lower one on the y-axes and vice-cersa for
Point2.
"""


def _compute_orientation(line):
    point1 = line[1]
    point2 = line[0]
    """
    If the latutide of a point is less and the longitude is more, or
    If the latitude of a point is more and the longitude is less, then
    the point is oriented leftward and wil have negative orientation.
    """
    if ((point2[0] > point1[0]) and (point2[1] < point1[1])) or (
        (point2[0] < point1[0]) and (point2[1] > point1[1])
    ):
        return -_compute_angle(point1, point2)
    # if the latitudes are same, the line is horizontal
    elif point2[1] == point1[1]:
        return 0
    # if the longitudes are same, the line is vertical
    elif point2[0] == point1[0]:
        return 90
    return _compute_angle(point1, point2)


"""
This below function calculates the acute joining angle between
two given set of points.
"""


def _points_set_angle(line1, line2):
    l1orien = _compute_orientation(line1)
    l2orien = _compute_orientation(line2)
    if ((l1orien > 0) and (l2orien < 0)) or ((l1orien < 0) and (l2orien > 0)):
        return abs(l1orien) + abs(l2orien)
    elif ((l1orien > 0) and (l2orien > 0)) or ((l1orien < 0) and (l2orien < 0)):
        theta1 = abs(l1orien) + 180 - abs(l2orien)
        theta2 = abs(l2orien) + 180 - abs(l1orien)
        if theta1 < theta2:
            return theta1
        else:
            return theta2
    elif (l1orien == 0) or (l2orien == 0):
        if l1orien < 0:
            return 180 - abs(l1orien)
        elif l2orien < 0:
            return 180 - abs(l2orien)
        else:
            return 180 - (
                abs(_compute_orientation(line1)) + abs(_compute_orientation(line2))
            )
    elif l1orien == l2orien:
        return 180


"""
The below function calculates the joining angle between
two line segments.
"""


def _angle_between_two_lines(line1, line2):
    l1p1, l1p2 = line1
    l2p1, l2p2 = line2
    l1orien = _compute_orientation(line1)
    l2orien = _compute_orientation(line2)
    """
    If both lines have same orientation, return 180 If one of the lines is zero,
    exception for that If both the lines are on same side of the horizontal plane,
    calculate 180-(sumOfOrientation) If both the lines are on same side of the vertical
    plane, calculate pointSetAngle
    """
    if l1orien == l2orien:
        angle = 180
    elif (l1orien == 0) or (l2orien == 0):
        angle = _points_set_angle(line1, line2)

    elif l1p1 == l2p1:
        if ((l1p1[1] > l1p2[1]) and (l1p1[1] > l2p2[1])) or (
            (l1p1[1] < l1p2[1]) and (l1p1[1] < l2p2[1])
        ):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = _points_set_angle([l1p1, l1p2], [l2p1, l2p2])
    elif l1p1 == l2p2:
        if ((l1p1[1] > l2p1[1]) and (l1p1[1] > l1p2[1])) or (
            (l1p1[1] < l2p1[1]) and (l1p1[1] < l1p2[1])
        ):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = _points_set_angle([l1p1, l1p2], [l2p2, l2p1])
    elif l1p2 == l2p1:
        if ((l1p2[1] > l1p1[1]) and (l1p2[1] > l2p2[1])) or (
            (l1p2[1] < l1p1[1]) and (l1p2[1] < l2p2[1])
        ):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = _points_set_angle([l1p2, l1p1], [l2p1, l2p2])
    elif l1p2 == l2p2:
        if ((l1p2[1] > l1p1[1]) and (l1p2[1] > l2p1[1])) or (
            (l1p2[1] < l1p1[1]) and (l1p2[1] < l2p1[1])
        ):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = _points_set_angle([l1p2, l1p1], [l2p2, l2p1])
    return angle


def _merge_lines_loop(n, unique_dict):
    outlist = set()
    current_edge1 = n

    outlist.add(current_edge1)

    while True:
        if (
            isinstance(unique_dict[current_edge1][6], int)
            and unique_dict[current_edge1][6] not in outlist
        ):
            current_edge1 = unique_dict[current_edge1][6]
            outlist.add(current_edge1)
        elif (
            isinstance(unique_dict[current_edge1][7], int)
            and unique_dict[current_edge1][7] not in outlist
        ):
            current_edge1 = unique_dict[current_edge1][7]
            outlist.add(current_edge1)
        else:
            break

    current_edge1 = n
    while True:
        if (
            isinstance(unique_dict[current_edge1][7], int)
            and unique_dict[current_edge1][7] not in outlist
        ):
            current_edge1 = unique_dict[current_edge1][7]
            outlist.add(current_edge1)
        elif (
            isinstance(unique_dict[current_edge1][6], int)
            and unique_dict[current_edge1][6] not in outlist
        ):
            current_edge1 = unique_dict[current_edge1][6]
            outlist.add(current_edge1)
        else:
            break

    outlist = list(outlist)
    outlist.sort()
    return outlist

if __name__ == '__main__':
    main()