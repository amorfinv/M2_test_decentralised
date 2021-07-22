import osmnx as ox
import networkx as nx
import graph_funcs
from os import path
from directionality import calcDirectionality
import math, multiprocessing
from functools import partial
import numpy as np
from shapely.geometry import Point, LineString, MultiLineString
from shapely import ops
import geopandas as gpd
import pandas as pd

# use osmnx environment here

def main():

    # working paths
    gis_data_path = path.join('gis')
    
    # convert shapefile to shapely polygon
    airspace_gdf = gpd.read_file(path.join(gis_data_path, 'small_network.gpkg'))
    airspace_poly = airspace_gdf.geometry.iloc[0]

    # create MultiDigraph from polygon
    G = ox.graph_from_polygon(airspace_poly, network_type='drive', simplify=True)
    
    # save as osmnx graph
    ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'raw_graph.gpkg'))
    
    # remove unconnected streets and add edge bearing attrbute 
    G = graph_funcs.remove_unconnected_streets(G)
    ox.add_edge_bearings(G)
    
    #ox.plot.plot_graph(G)
    
    # # manually remove some nodes and edges to clean up graph
    # nodes_to_remove = [33144419, 33182073, 33344823, 83345330, 33345337,
    #                    33345330, 33344804, 30696018, 5316966255, 5316966252, 33344821,
    #                    2335589819, 245498397, 287914700, 271016303, 2451285012, 393097]
    # edges_to_remove = [(291088171, 3155094143), (60957703, 287914700), (2451285012, 287914700),
    #                    (25280685, 30696019), (30696019, 25280685), (392251, 25280685), 
    #                    (25280685, 392251), (33301346, 1119870220),  
    #                    (33345331, 33345333), (378699, 378696), (378696, 33143911), 
    #                    (33143911, 33144821), (264061926, 264055537), (33144706, 33144712),
    #                    (33144712, 33174086), (33174086, 33144719), (33144719, 92739749),
    #                    (33345319, 29048469), (287914700, 60957703), (213287623, 251207325),
    #                    (251207325, 213287623)]
    # G.remove_nodes_from(nodes_to_remove)
    # G.remove_edges_from(edges_to_remove)
    
    #ox.plot.plot_graph(G)
    
    # convert graph to geodataframe
    g = ox.graph_to_gdfs(G)
    
    # # get node and edge geodataframe
    nodes = g[0]
    edges = g[1]
    
    # remove double two way edges
    edges = graph_funcs.remove_two_way_edges(edges)
    
    # remove non parallel opposite edges (or long way)
    edges = graph_funcs.remove_long_way_edges(edges)
    
    # allocated edge height based on cardinal method
    layer_allocation, _ = graph_funcs.allocate_edge_height(edges, 0)
    
    # Assign layer allocation to geodataframe
    edges['layer_height'] = layer_allocation
    
    # # add interior angles at all intersections
    edges = graph_funcs.add_edge_interior_angles(edges)
    
    # Perform COINS algorithm to add stroke groups
    coins_obj = COINS(edges)
    edges['stroke_group'] = coins_obj.stroke_attribute()
    group_gdf = coins_obj.stroke_gdf()
    
    init_edge_directions = graph_funcs.get_first_group_edges(G, group_gdf, edges)
    
    # Apply direction algorithm
    # #edge_directions = calcDirectionality(group_gdf, nodes, init_edge_directions)
    # edge_directions = [(33302019, 378727, 0),   # group 0
    #             (33144416, 33144414, 0),    # group 1
    #             (30696015, 64975746, 0),    # group 2
    #             (378728, 33345331, 0),      # group 3
    #             (60631071, 401838, 0),      # group 4
    #             (264055540, 33144941, 0),   # group 5
    #             (33345285, 33345298, 0),    # group 6
    #             (251523470, 33345297, 0),   # group 7
    #             (33144366, 33143888, 0),    # group 8
    #             (1119870220, 394751, 0),    # group 9
    #             (378696, 3312560802, 0),    # group 10
    #             (64975949, 33345310, 0),    # group 11
    #             (3155094143, 64971266, 0),  # group 12
    #             (33144706, 33144601, 0),    # group 13
    #             (33344824, 33344825, 0),    # group 14
    #             (33344807, 33144550, 0),    # group 15
    #             (655012, 33144500, 0),      # group 16
    #             (33345286, 33345303, 0),    # group 17
    #             (283324403, 358517297, 0),  # group 18
    #             (33344802, 33344805, 0),    # group 19
    #             (264055537, 264055538, 0),  # group 20
    #             (29048460, 33345320, 0),    # group 21
    #             (33144712, 33144605, 0),    # group 22
    #             (33143911, 33143898, 0),    # group 23
    #             (29048469, 64972028, 0),    # group 24
    #             (64975551, 33345319, 0),    # group 25
    #             (92739749, 33144621, 0),    # group 26
    #             (33144633, 33144941, 0),    # group 27
    #             (33144560, 283324407, 0),   # group 28
    #             (25280685, 33344817, 0),    # group 29
    #             (33144566, 33144555, 0),    # group 30
    #             (33345332, 33345333, 0),    # group 31
    #             (33144471, 33144422, 0),    # group 32
    #             (33144659, 33144655, 0),    # group 33
    #             (33144719, 33144616, 0),    # group 34
    #             (33344808, 33144550, 0),    # group 35
    #             (33344812, 33344811, 0),    # group 36
    #             (245498401, 245498398, 0),  # group 37 
    #             (33144637, 320192043, 0),   # group 38 
    #             (33144755, 33144759, 0),    # group 39 
    #             (33344809, 2423479559, 0),  # group 40 
    #             (33344816, 392251, 0),      # group 41 
    #             (33345310, 33345289, 0),    # group 42 
    #             (33345299, 33344825, 0),    # group 43 
    #             (33345321, 33345291, 0),    # group 44
    #             (64975131, 60957703, 0)     # group 45 
    #             ]
    
    # # reoroder edge geodatframe
    # edges = graph_funcs.set_direction(edges, edge_directions)
    
    # create graph and save edited
    G = ox.graph_from_gdfs(nodes, edges)
    
    # save as osmnx graph
    ox.save_graphml(G, filepath=path.join(gis_data_path, 'processed_graph.graphml'))
    
    # Save geopackage for import to QGIS and momepy
    ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'processed_graph.gpkg'))
    
    # save csv for reference
    edges.to_csv(path.join(gis_data_path, 'edges.csv'))
    nodes.to_csv(path.join(gis_data_path, 'nodes.csv'))
    

#--------------------------- COINS -------------------------
class COINS:
    
    """
    Calculates natural continuity and hierarchy of street networks in given GeoDataFrame.
    with COINS algorithm. Creates 'strokes', refer to journal paper for more details.

    Parameters
    ----------
    edge_gdf : GeoDataFrame
        GeoDataFrame containing edge geometry of street network
    angle_threshold : int, float (default 0)
        the angle threshold for COINS algorithm. Segments will only be considered 
        a part of the same street if deflection angle is above the threshold.

    Returns
    ----------
    - series containing the stroke_group.
    - new GeoDataFrame prior to performing merging. Refer to COINS paper for more details.
    - new GeoDataFrame after merging. Refer to COINS paper for more details.

    Examples
    --------

    >>> coins = momepy.COINS(streets)

    >>> premerge = coins.premerge() 

    >>> stroke_gdf = coins.stroke_gdf()  

    >>> stroke_attr = coins.stroke_attribute() 
    
    """

    def __init__(self, edge_gdf, angle_threshold=0):
        self.edge_gdf = edge_gdf
        self.gdfProjection = self.edge_gdf.crs
        self.already_merged = False

        # Get indices of original gdf
        self.uv_index =  self.edge_gdf.index.tolist()

        # Get line segments from edge gdf
        self.lines = [list(value[1].coords) for value in edge_gdf['geometry'].iteritems()]

        # split edges into line segments
        self.splitLines()

        # create unique_id for each individual line segment
        self.uniqueID()

        # Compute edge connectivity table
        self.getLinks()

        # Find best link at every point for both lines
        self.bestLink()

        # Cross check best links and enter angle threshold for connectivity
        self.crossCheckLinks(angle_threshold)

    def premerge(self):
        return self.create_gdf_premerge()

    def stroke_gdf(self):
        if not self.already_merged: 
            self.mergeLines()
        return self.create_gdf_strokes()

    def stroke_attribute(self):
        if not self.already_merged:
            self.mergeLines()
        return self.add_gdf_stroke_attributes()

    def splitLines(self):
        outLine = []
        self.tempArray = []
        n = 0
        #Iterate through the lines and split the edges
        idx = 0
        for line in self.lines:
            for part in _listToPairs(line):
                outLine.append([part, _computeOrientation(part), list(), list(), list(), list(), list(), list(), self.uv_index[idx]])
                # Merge the coordinates as string, this will help in finding adjacent edges in the function below
                self.tempArray.append([n, f'{part[0][0]}_{part[0][1]}', f'{part[1][0]}_{part[1][1]}'])
                n += 1
            idx += 1

        self.split = outLine

    def uniqueID(self):
    #Loop through split lines, assign unique ID and
    #store inside a list along with the connectivity dictionary
        self.unique = dict(enumerate(self.split))

    def getLinks(self):
        print("Finding adjacent segments...")

        self.tempArray = np.array(self.tempArray, dtype=object)
        
        iterations = [n for n in range(0,len(self.unique))]
        
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        constantParameterFunction = partial(_getLinksMultiprocessing, total=len(self.unique), tempArray=self.tempArray)
        self.result = pool.map(constantParameterFunction, iterations)
        pool.close()
        pool.join()
        iterations = None

        for a in self.result:
            n = a[0]
            self.unique[n][2] = a[1]
            self.unique[n][3] = a[2]
            
        print('>'*50 + ' [%d/%d] '%(len(self.unique),len(self.unique)) + '100%' + '\n', end='\r')
            
    def bestLink(self):
        self.anglePairs = dict()
        for edge in range(0,len(self.unique)):
            p1AngleSet = []
            p2AngleSet = []

            """
            Instead of computing the angle between the two segments twice, the method calculates
            it once and stores in the dictionary for both the keys. So that it does not calculate
            the second time because the key is already present in the dictionary.
            """
            for link1 in self.unique[edge][2]:
                self.anglePairs["%d_%d" % (edge, link1)] = _angleBetweenTwoLines(self.unique[edge][0], self.unique[link1][0])
                p1AngleSet.append(self.anglePairs["%d_%d" % (edge, link1)])
                
            for link2 in self.unique[edge][3]:
                self.anglePairs["%d_%d" % (edge, link2)] = _angleBetweenTwoLines(self.unique[edge][0], self.unique[link2][0])
                p2AngleSet.append(self.anglePairs["%d_%d" % (edge, link2)])

            """
            Among the adjacent segments deflection angle values, check for the maximum value
            at both the ends. The segment with the maximum angle is stored in the attributes
            to be cross-checked later for before finalising the segments at both the ends.
            """
            if len(p1AngleSet)!=0:
                val1, idx1 = max((val, idx) for (idx, val) in enumerate(p1AngleSet))
                self.unique[edge][4] = self.unique[edge][2][idx1], val1
            else:
                self.unique[edge][4] = 'DeadEnd'
                
            if len(p2AngleSet)!=0:
                val2, idx2 = max((val, idx) for (idx, val) in enumerate(p2AngleSet))
                self.unique[edge][5] = self.unique[edge][3][idx2], val2
            else:
                self.unique[edge][5] = 'DeadEnd'

    def crossCheckLinks(self, angleThreshold):
        print("Cross-checking and finalising the links...")
        for edge in range(0,len(self.unique)):
            # Printing the progress bar
            if edge%1000==0:
                """
                Dividing by two to have 50 progress steps
                Subtracting from 50, and not hundred to have less progress steps
                """
                currentProgress = math.floor(100*edge/len(self.unique)/2)
                remainingProgress = 50 - currentProgress            
                print('>'*currentProgress + '-' * remainingProgress + ' [%d/%d] '%(edge,len(self.unique)) + '%d%%'%(currentProgress*2), end='\r')

            bestP1 = self.unique[edge][4][0]
            bestP2 = self.unique[edge][5][0]
            
            if type(bestP1) == type(1) and \
               edge in [self.unique[bestP1][4][0], self.unique[bestP1][5][0]] and \
               self.anglePairs["%d_%d" % (edge, bestP1)] > angleThreshold:
                self.unique[edge][6] = bestP1
            else:
                self.unique[edge][6] = 'LineBreak'
                
            if type(bestP2) == type(1) and \
               edge in [self.unique[bestP2][4][0], self.unique[bestP2][5][0]] and \
               self.anglePairs["%d_%d" % (edge, bestP2)] > angleThreshold:
                self.unique[edge][7] = bestP2
            else:
                self.unique[edge][7] = 'LineBreak'
                
        print('>'*50 + ' [%d/%d] '%(edge+1,len(self.unique)) + '100%' + '\n', end='\r')

    def addLine(self, edge, parent=None, child='Undefined'):
        if child=='Undefined':
            self.mainEdge = len(self.merged)
        if not edge in self.assignedList:
            if parent==None:
                currentid = len(self.merged)
                self.merged[currentid] = set()
            else:
                currentid = self.mainEdge
            self.merged[currentid].add(_listToTuple(self.unique[edge][0]))
            self.assignedList.append(edge)
            link1 = self.unique[edge][6]
            link2 = self.unique[edge][7]
            if type(1) == type(link1):
                self.addLine(link1, parent=edge, child=self.mainEdge)
            if type(1) == type(link2):
                self.addLine(link2, parent=edge, child=self.mainEdge)

    def mergeLines(self):
        print('Merging Lines...')
        self.mergingList = list()
        self.merged = list()
        self.edge_idx = list()

        iterations = [n for n in range(0,len(self.unique))]
        
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        constantParameterFunction = partial(_mergeLinesMultiprocessing, total=len(self.unique), uniqueDict=self.unique)
        self.result = pool.map(constantParameterFunction, iterations)
        pool.close()
        pool.join()
        iterations = None

        for tempList in self.result:
            if not tempList in self.mergingList:
                self.mergingList.append(tempList)
                self.merged.append({_listToTuple(self.unique[key][0]) for key in tempList})
                
                # assign stroke number to edge from argument
                self.edge_idx.append({self.unique[key][8] for key in tempList})
            
        self.merged = dict(enumerate(self.merged))
        self.edge_idx = dict(enumerate(self.edge_idx))
        self.already_merged = True
        print('>'*50 + ' [%d/%d] '%(len(self.unique),len(self.unique)) + '100%' + '\n', end='\r')
        
    #Export geodataframes, 3 options
    def create_gdf_premerge(self):
        # create empty list to fill out
        myList = []
        
        for parts in range(0,len(self.unique)):
            # get all segment points and make line
            lineList = _tupleToList(self.unique[parts][0])
            geom_line = LineString([Point(lineList[0]), Point(lineList[1])])
            
            # get other values for premerged
            UniqueID = parts 
            Orientation = self.unique[parts][1]
            linksP1 = self.unique[parts][2]
            linksP2 = self.unique[parts][3]
            bestP1 = self.unique[parts][4]
            bestP2 = self.unique[parts][5]
            P1Final = self.unique[parts][6]
            P2Final = self.unique[parts][7]
            
            # append list
            myList.append([UniqueID, Orientation, linksP1, linksP2, bestP1, bestP2, P1Final, P2Final, geom_line])

        edge_gdf = gpd.GeoDataFrame(myList, columns=['UniqueID','Orientation','linksP1','linksP2','bestP1','bestP2','P1Final','P2Final', 'geometry'], crs=self.gdfProjection)
        edge_gdf.set_index('UniqueID', inplace=True)

        return edge_gdf

    def create_gdf_strokes(self):

        # create empty list to fill 
        myList = []
        
        # loop through merged geometry
        for a in self.merged:

            # get all segment points and make line strings
            linelist = _tupleToList(list(self.merged[a]))
            list_lines_segments = []

            for b in linelist:
                list_lines_segments.append(LineString(b))      
            
            # merge seperate segments
            geom_multi_line = ops.linemerge(MultiLineString(list_lines_segments))

            # get other values for gdf
            ID_value = a 
            nSegments = len(self.merged[a])

            # append list
            myList.append([ID_value, nSegments, geom_multi_line])

        edge_gdf = gpd.GeoDataFrame(myList, columns=['stroke_group', 'nSegments', 'geometry'], crs=self.gdfProjection)
        edge_gdf.set_index('stroke_group', inplace=True)
        
        return edge_gdf

    def add_gdf_stroke_attributes(self):

        # Invert self.edge_idx to get a dictionary where the key is the original edge index and the value is the group
        inv_edges = {value: key for key in self.edge_idx for value in self.edge_idx[key]}

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
def _tupleToList(line):
    for a in range(0,len(line)):
        line[a] = list(line[a])
    return(line)

def _listToTuple(line):
    for a in range(0, len(line)):
        line[a] = tuple(line[a])
    return(tuple(line))
"""
The below function takes a line as an input and splits
it at every point.
"""
def _listToPairs(inList):
    outList = []
    index = 0
    for index in range(0,len(inList)-1):
        tempList = [list(inList[index]), list(inList[index+1])]
        outList.append(tempList)
    return(outList)

"""
The function below calculates the angle between two points in space.
"""

def _computeAngle(point1, point2):
    height = abs(point2[1] - point1[1])
    base = abs(point2[0] - point1[0])
    angle = round(math.degrees(math.atan(height/base)), 3)
    return(angle)

"""
This function calculates the orientation of a line segment.
Point1 is the lower one on the y-axes and vice-cersa for
Point2.
"""
def _computeOrientation(line):
    point1 = line[1]
    point2 = line[0]
    """
    If the latutide of a point is less and the longitude is more, or
    If the latitude of a point is more and the longitude is less, then
    the point is oriented leftward and wil have negative orientation.
    """
    if ((point2[0] > point1[0]) and (point2[1] < point1[1])) or ((point2[0] < point1[0]) and (point2[1] > point1[1])):
        return(-_computeAngle(point1, point2))
    #If the latitudes are same, the line is horizontal
    elif point2[1] == point1[1]:
        return(0)
    #If the longitudes are same, the line is vertical
    elif point2[0] == point1[0]:
        return(90)
    else:
        return(_computeAngle(point1, point2))

"""
This below function calculates the acute joining angle between
two given set of points.
"""
def _pointsSetAngle(line1, line2):
    l1orien = _computeOrientation(line1)
    l2orien = _computeOrientation(line2)
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
            return(180 - (abs(_computeOrientation(line1)) + abs(_computeOrientation(line2))))
    elif (l1orien==l2orien):
        return(180)
        
"""
The below function calculates the joining angle between
two line segments.
"""
def _angleBetweenTwoLines(line1, line2):
    l1p1, l1p2 = line1
    l2p1, l2p2 = line2
    l1orien = _computeOrientation(line1)
    l2orien = _computeOrientation(line2)
    """
    If both lines have same orientation, return 180
    If one of the lines is zero, exception for that
    If both the lines are on same side of the horizontal plane, calculate 180-(sumOfOrientation)
    If both the lines are on same side of the vertical plane, calculate pointSetAngle
    """
    if (l1orien==l2orien): 
        angle = 180
    elif (l1orien==0) or (l2orien==0): 
        angle = _pointsSetAngle(line1, line2)
        
    elif l1p1 == l2p1:
        if ((l1p1[1] > l1p2[1]) and (l1p1[1] > l2p2[1])) or ((l1p1[1] < l1p2[1]) and (l1p1[1] < l2p2[1])):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = _pointsSetAngle([l1p1, l1p2], [l2p1,l2p2])
    elif l1p1 == l2p2:
        if ((l1p1[1] > l2p1[1]) and (l1p1[1] > l1p2[1])) or ((l1p1[1] < l2p1[1]) and (l1p1[1] < l1p2[1])):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = _pointsSetAngle([l1p1, l1p2], [l2p2,l2p1])
    elif l1p2 == l2p1:
        if ((l1p2[1] > l1p1[1]) and (l1p2[1] > l2p2[1])) or ((l1p2[1] < l1p1[1]) and (l1p2[1] < l2p2[1])):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = _pointsSetAngle([l1p2, l1p1], [l2p1,l2p2])
    elif l1p2 == l2p2:
        if ((l1p2[1] > l1p1[1]) and (l1p2[1] > l2p1[1])) or ((l1p2[1] < l1p1[1]) and (l1p2[1] < l2p1[1])):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = _pointsSetAngle([l1p2, l1p1], [l2p2,l2p1])
    return(angle)

def _getLinksMultiprocessing(n, total, tempArray):
    # Printing the progress bar
    if n%1000==0:
        """
        Dividing by two to have 50 progress steps
        Subtracting from 50, and not hundred to have less progress steps
        """
        currentProgress = math.floor(100*n/total/2)
        remainingProgress = 50 - currentProgress            
        print('>'*currentProgress + '-' * remainingProgress + ' [%d/%d] '%(n,total) + '%d%%'%(currentProgress*2), end='\r')

    # Create mask for adjacent edges as endpoint 1
    m1 = tempArray[:,1]==tempArray[n,1]
    m2 = tempArray[:,2]==tempArray[n,1]
    mask1 = m1 + m2

    # Create mask for adjacent edges as endpoint 2
    m1 = tempArray[:,1]==tempArray[n,2]
    m2 = tempArray[:,2]==tempArray[n,2]
    mask2 = m1 + m2

    # Use the tempArray to extract only the uniqueIDs of the adjacent edges at both ends
    mask1 = tempArray[:,0][~(mask1==0)]
    mask2 = tempArray[:,0][~(mask2==0)]

    # Links (excluding the segment itself) at both the ends are converted to list and added to the 'unique' attribute
    return(n, list(mask1[mask1 != n]), list(mask2[mask2 != n]))

def _mergeLinesMultiprocessing(n, total, uniqueDict):
    # Printing the progress bar
    if n%1000==0:
        """
        Dividing by two to have 50 progress steps
        Subtracting from 50, and not hundred to have less progress steps
        """
        currentProgress = math.floor(100*n/total/2)
        remainingProgress = 50 - currentProgress            
        print('>'*currentProgress + '-' * remainingProgress + ' [%d/%d] '%(n,total) + '%d%%'%(currentProgress*2), end='\r')
        
    outlist = set()
    currentEdge1 = n

    outlist.add(currentEdge1)

    while True:
        if type(uniqueDict[currentEdge1][6]) == type(1) and \
           uniqueDict[currentEdge1][6] not in outlist:
            currentEdge1 = uniqueDict[currentEdge1][6]
            outlist.add(currentEdge1)
        elif type(uniqueDict[currentEdge1][7]) == type(1) and \
           uniqueDict[currentEdge1][7] not in outlist:
            currentEdge1 = uniqueDict[currentEdge1][7]
            outlist.add(currentEdge1)
        else:
            break
    currentEdge1 = n
    while True:
        if type(uniqueDict[currentEdge1][7]) == type(1) and \
           uniqueDict[currentEdge1][7] not in outlist:
            currentEdge1 = uniqueDict[currentEdge1][7]
            outlist.add(currentEdge1)
        elif type(uniqueDict[currentEdge1][6]) == type(1) and \
           uniqueDict[currentEdge1][6] not in outlist:
            currentEdge1 = uniqueDict[currentEdge1][6]
            outlist.add(currentEdge1)
        else:
            break

    outlist = list(outlist)
    outlist.sort()
    return(outlist)
#----------------------END OF COINS----------------------------

if __name__ == '__main__':
    main()