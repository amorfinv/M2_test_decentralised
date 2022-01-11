# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:51:03 2021

@author: nipat
"""
###############################
#This values made get_path to stack!!!!!!!!!!!!!!!!!
#origin=[16.342228782395118,48.263495892875554]
#destination=[16.381609676387115,48.16806791791769  ]##
##############################

import heapq
import numpy as np
import math
import copy
from shapely.geometry import Point
from pyproj import  Transformer
import shapely
from plugins.streets.flow_control import street_graph,bbox
from plugins.streets.open_airspace_grid import Cell, open_airspace

##########
##Functions for finding the cell of a point in open airspace
def index_2d(myList, v):
    test=[]
    for i in range(len(myList)):
        test.append(myList[i][1])

    return (test.index(v))


def check_where(p,min_x_ind,cells):
    x=p[0]
    y=p[1]

    many=[i for i in min_x_ind if i[0] <= x and i[1]>=x] 
    which=[]
    for i in range(len(many)):
        which.append(many[i][2])
 
    canditade_cells=[]
    for i in which:
        canditade_cells.append(cells[index_2d(cells,min_x_ind[i][3])])

    wh=-1
    for i in range(len(canditade_cells)):
        mn=np.argmin(canditade_cells[i][0], axis=0)
        mx=np.argmax(canditade_cells[i][0], axis=0)

        if canditade_cells[i][0][mn[1]][1] <=y and canditade_cells[i][0][mx[1]][1]>=y : 
            
            wh=i
            break
    if wh !=-1:
        winner=canditade_cells[wh]
    else :
        winner=-1

    return winner
    
def sort_cells_x(cells):
    min_x_ind=[]
    for i in range(len(cells)):
        mn=np.argmin(cells[i][0], axis=0)
        mx=np.argmax(cells[i][0], axis=0)
        min_x_ind.append([cells[i][0][mn[0]][0],cells[i][0][mx[0]][0],i,cells[i][1]])
        
    #print(sorted(min_x_ind,key=lambda x: (x[1])))
    return min_x_ind
##########################

# Given three collinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def onSegment(p, q, r):
    if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
            (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
        return True
    return False  
                
def ccw(A,B,C):
    if (C[1]-A[1]) * (B[0]-A[0]) == (B[1]-A[1]) * (C[0]-A[0]):
        return 2
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    if ccw(A,B,C)==2 and onSegment(A,C,B):
        return True
    elif ccw(A,B,D)==2 and onSegment(A,D,B):
        return True
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def line_intersection_point(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return   None,None#raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def distance_to_line(A, B, E) :
    # vector AB
    AB = [None, None]
    AB[0] = B[0] - A[0]
    AB[1] = B[1] - A[1]
 
    # vector BP
    BE = [None, None]
    BE[0] = E[0] - B[0]
    BE[1] = E[1] - B[1]
 
    # vector AP
    AE = [None, None];
    AE[0] = E[0] - A[0]
    AE[1] = E[1] - A[1]
 
    # Variables to store dot product
 
    # Calculating the dot product
    AB_BE = AB[0] * BE[0] + AB[1] * BE[1]
    AB_AE = AB[0] * AE[0] + AB[1] * AE[1]
 
    # Minimum distance from
    # point E to the line segment
    reqAns = 0
 
    # Case 1
    if (AB_BE > 0) :
 
        # Finding the magnitude
        y = E[1] - B[1]
        x = E[0] - B[0]
        reqAns = math.sqrt(x * x + y * y)

    # Case 2
    elif (AB_AE < 0) :
        y = E[1] - A[1]
        x = E[0] - A[0]
        reqAns = math.sqrt(x * x + y * y)
 
    # Case 3
    else:
 
        # Finding the perpendicular distance
        x1 = AB[0]
        y1 = AB[1]
        x2 = AE[0]
        y2 = AE[1]
        mod = math.sqrt(x1 * x1 + y1 * y1)
        reqAns = abs(x1 * y2 - y1 * x2) / mod
     
    return reqAns
 
def find_closest_point_on_linesegment(line,point):
    ##https://diego.assencio.com/?index=ec3d5dfdfc0b6a0d147a656f0af332bd
    
    pp=[0,0]
    ls=((point[0]-line[0][0])*(line[1][0]-line[0][0])+(point[1]-line[0][1])*(line[1][1]-line[0][1]))/((line[1][0]-line[0][0])*(line[1][0]-line[0][0])+(line[1][1]-line[0][1])*(line[1][1]-line[0][1]))
    if ls<=0:
        pp=line[0]
    elif ls>=1:
        pp=line[1]
    else:
        pp[0]=line[0][0]+ls*(line[1][0]-line[0][0])
        pp[1]=line[0][1]+ls*(line[1][1]-line[0][1])
        
    return pp

def distance_point(A,B):
    R = 6372800  # Earth radius in meters
    lat1=A[1]
    lon1=A[0]
    lat2=B[1]
    lon2=B[0]
        
    phi1, phi2 = math.radians(lat1), math.radians(lat2) 
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
        
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))       
  
##Check if point A lies between points B and C      
def lies_between(A,B,C):
    #https://stackoverflow.com/questions/33155240/how-to-check-if-a-point-is-between-two-other-points-but-not-limited-to-be-align?rq=1
    a = distance_point(B,C)
    b = distance_point(C,A)
    c = distance_point(A,B)
    return a**2 + b**2 >= c**2 and a**2 + c**2 >= b**2

##Returns the nearest edge of a point
def get_nearest_edge(gdf, point):
    """
    Return the nearest edge to a pair of coordinates. Pass in a graph and a tuple
    with the coordinates. We first get all the edges in the graph. Secondly we compute
    the euclidean distance from the coordinates to the segments determined by each edge.
    The last step is to sort the edge segments in ascending order based on the distance
    from the coordinates to the edge. In the end, the first element in the list of edges
    will be the closest edge that we will return as a tuple containing the shapely
    geometry and the u, v nodes.
    Parameters
    ----------
    G : networkx multidigraph
    point : tuple
        The (lat, lng) or (y, x) point for which we will find the nearest edge
        in the graph
    Returns
    -------
    closest_edge_to_point : tuple (shapely.geometry, u, v)
        A geometry object representing the segment and the coordinates of the two
        nodes that determine the edge section, u and v, the OSM ids of the nodes.
    """

    graph_edges = gdf[["geometry"]].values.tolist()
    graph_edges_indexes=gdf.index.tolist()
    for i in range(len(graph_edges)):# maybe do that faster?
        graph_edges[i].append(graph_edges_indexes[i][0])
        graph_edges[i].append(graph_edges_indexes[i][1])


    edges_with_distances = [
        (
            graph_edge,
            Point(tuple(reversed(point))).distance(graph_edge[0])
        )
        for graph_edge in graph_edges
    ]

    edges_with_distances = sorted(edges_with_distances, key=lambda x: x[1])
    closest_edge_to_point = edges_with_distances[0][0]
    #print("distance",edges_with_distances[0][1])

    geometry, u, v = closest_edge_to_point
    distance=edges_with_distances[0][1]


    return geometry, u, v,distance


# =============================================================================
# class CellNode:
#     def __init__(self,cell):
#         self.p0=cell.p0
#         self.p1=cell.p1
#         self.p2=cell.p2
#         self.p3=cell.p3
# 
# class Node:
#     av_speed_horizontal= 10.0#10.0 ##TODO: that needs fine tunng
#     
#     def __init__(self,key_index,index,group):
#         self.key_index=key_index # the index the osmnx graph
#         self.index=index# the index in the search graph   # do that numpy uint16
# 
#         #the coordinates of the node as given by osmnx (latitude,longitude)
#         ##Coordinates of the center
# # =============================================================================
# #         self.lon=None
# #         self.lat=None
# # 
# #         
# #         ##Coordinates of the center
# #         self.x_cartesian=None
# #         self.y_cartesian=None
# # =============================================================================
# 
# 
#         #the parents(predessecors) and children(successor) of the node expressed as lists containing their indexes in the graph 
#         self.parents=[]
#         self.children=[]
#         
#         #self.f=0.0
#         self.g=float('inf')
#         self.rhs=float('inf')
#         self.key=[0.0,0.0]
#  
#         self.inQueue=False #do that numpy bool8
#         
#         #the stroke group
#         self.group=group # do that numpy uint16
#         
#         self.expanded=False #do that numpy bool8
#         
#         #self.open_airspace=False
#         #self.cell=None
# =============================================================================
##from geo.py
def rwgs84(latd):
    """ Calculate the earths radius with WGS'84 geoid definition
        In:  lat [deg] (latitude)
        Out: R   [m]   (earth radius) """
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

##from geo.py
def qdrdist(latd1, lond1, latd2, lond2):
    """ Calculate bearing and distance, using WGS'84
        In:
            latd1,lond1 en latd2, lond2 [deg] :positions 1 & 2
        Out:
            qdr [deg] = heading from 1 to 2
            d [nm]    = distance from 1 to 2 in nm """

    # Haversine with average radius for direction

    # Constants
    nm  = 1852.  # m       1 nautical mile

    # Check for hemisphere crossing,
    # when simple average would not work

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

    #root = sin1 * sin1 + coslat1 * coslat2 * sin2 * sin2
    #d    =  2.0 * r * np.arctan2(np.sqrt(root) , np.sqrt(1.0 - root))
    # d =2.*r*np.arcsin(np.sqrt(sin1*sin1 + coslat1*coslat2*sin2*sin2))

    # Corrected to avoid "nan" at westward direction
    d = r*np.arccos(np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1) + \
                 np.sin(lat1)*np.sin(lat2))

    # Bearing from Ref. http://www.movable-type.co.uk/scripts/latlong.html

    # sin1 = np.sin(0.5 * (lat2 - lat1))
    # sin2 = np.sin(0.5 * (lon2 - lon1))

    coslat1 = np.cos(lat1)
    coslat2 = np.cos(lat2)

    qdr = np.degrees(np.arctan2(np.sin(lon2 - lon1) * coslat2,
                                coslat1 * np.sin(lat2) -
                                np.sin(lat1) * coslat2 * np.cos(lon2 - lon1)))

    return qdr, d/nm

        
class Path:
    def __init__(self,start,goal,speed,graph):
        self.start=start
        self.goal=goal
        self.k_m=0
        self.queue=[]
        self.origin_node_index=None
        self.speed=speed
        self.graph=graph
        
##Initialise the path planning      
def initialise(path,flow_graph):
    path.queue=[]
    path.k_m=0
    path.graph.rhs_list[path.goal]=0  #path.goal.rhs=0
    path.graph.inQueue_list[path.goal]=True #path.goal.inQueue=True
    h=heuristic(path.start,path.goal,path.speed,flow_graph,path.graph) #path.goal.h=heuristic(path.start,path.goal,path.speed,flow_graph)
    path.graph.expanded_list[path.goal]=True   #path.goal.expanded=True
    heapq.heappush(path.queue, (h,0,path.goal))
    path.origin_node_index=path.start
 

##Compare the keys of two nodes
def compare_keys(key1,key2):
    if key1[0]<key2[0]:
        return True
    elif key1[0]==key2[0] and key1[1]<key2[1]:
        return True
    return False
    
##Calculate the keys of a node    
def calculateKey(node,start, path,flow_graph,graph):

    return (min(graph.g_list[node], graph.rhs_list[node]) + heuristic(node,start,path.speed,flow_graph,graph) + path.k_m, min(graph.g_list[node], graph.rhs_list[node]))
    #return (min(node.g, node.rhs) + heuristic(node,start,path.speed,flow_graph) + path.k_m, min(node.g, node.rhs))

##Calculate the distance of two points in cartesian coordinates
def eucledean_distance(p1,p2):
    return  math.sqrt((p1.x_cartesian-p2.x_cartesian)*(p1.x_cartesian-p2.x_cartesian)+ (p1.y_cartesian-p2.y_cartesian)*(p1.y_cartesian-p2.y_cartesian) )    

def heuristic(current, goal,speed,flow_graph,graph):
    cc=flow_graph.nodes_graph[graph.key_indices_list[current]]
    gg=flow_graph.nodes_graph[graph.key_indices_list[goal]]
    av_speed_vertical=1.0
    if cc.open_airspace or gg.open_airspace:
        h=eucledean_distance(cc, gg)/speed
    else:
        h=(abs(cc.x_cartesian-gg.x_cartesian)+abs(cc.y_cartesian-gg.y_cartesian))/speed

    if graph.groups_list[current]!=graph.groups_list[goal]:
        h=h+9.144/av_speed_vertical
    return h


##Compute the cost of moving from a node to its neighborh node
def compute_c(current,neigh,edges_speed,flow_graph,speed,graph):
    av_speed_vertical=1.0
    g=1
    cc=flow_graph.nodes_graph[graph.key_indices_list[current]]
    nn=flow_graph.nodes_graph[graph.key_indices_list[neigh]]
    
    if cc.open_airspace  or nn.open_airspace:
        g=eucledean_distance(cc,nn)/speed
    else:
        if graph.groups_list[current]!=graph.groups_list[neigh]:
            g=9.144/av_speed_vertical
        else:
            #check if the group is changing (the drone needs to turn)
            if graph.groups_list[current]==graph.groups_list[neigh]:
                if edges_speed[graph.key_indices_list[current]][graph.key_indices_list[neigh]]==0:
                    g=float('inf')
    
                else:
                    g=flow_graph.edges_graph[graph.key_indices_list[current]][graph.key_indices_list[neigh]].length/min(edges_speed[graph.key_indices_list[current]][graph.key_indices_list[neigh]],speed)
    return g    

##Return the top key of the queue 
def top_key(q):
    if len(q) > 0:
        return [q[0][0],q[0][1]]
    else:
        print('empty queue!')
        return [float('inf'), float('inf')]
    
##Update the vertex of a node
def update_vertex(path,node,flow_graph,graph):

    if graph.g_list[node]!=graph.rhs_list[node] and graph.inQueue_list[node]:     
        #Update
        id_in_queue = [item for item in path.queue if node==item[2]]
        if id_in_queue != []:
            if len(id_in_queue) != 1:
                raise ValueError('more than one ' + str(node) + ' in the queue!')
            graph.key_list[node]=calculateKey(node, path.start, path,flow_graph,graph)
            path.queue[path.queue.index(id_in_queue[0])]=path.queue[-1]
            path.queue.pop()
            heapq.heapify(path.queue)
            heapq.heappush(path.queue, (graph.key_list[node][0],graph.key_list[node][1],node))
            
    elif graph.g_list[node]!=graph.rhs_list[node] and (not graph.inQueue_list[node]):
        #Insert
        graph.inQueue_list[node]=True
        graph.key_list[node]=calculateKey(node, path.start, path,flow_graph,graph)
        heapq.heappush(path.queue, (graph.key_list[node][0],graph.key_list[node][1],node))
        
    elif graph.g_list[node]==graph.rhs_list[node] and graph.inQueue_list[node]: 
        #remove
        id_in_queue = [item for item in path.queue if node==item[2]]
        
        if id_in_queue != []:
            if len(id_in_queue) != 1:
                raise ValueError('more than one ' + id + ' in the queue!')
            graph.inQueue_list[node]=False
            path.queue[path.queue.index(id_in_queue[0])]=path.queue[-1]
            path.queue.pop()
            heapq.heapify(path.queue)
          

          
##Compute the shortest path using D* Lite
##returns flase if no path was found
def compute_shortest_path(path,graph,edges_speed,flow_graph):

    graph.key_list[path.start]=calculateKey(path.start, path.start, path,flow_graph,graph)
    k_old=top_key(path.queue)
   
    while graph.rhs_list[path.start]>graph.g_list[path.start] or compare_keys(k_old,graph.key_list[path.start]):

        if len(path.queue)==0:
            print("No path found!")
            return 0

        k_old=top_key(path.queue)
        current_node=path.queue[0][2]#get the node with the highest priority
        graph.expanded_list[current_node]=True

        
        k_new=calculateKey(current_node, path.start, path,flow_graph,graph)
        
        if compare_keys(k_old, k_new):
            heapq.heappop(path.queue)
            graph.key_list[current_node]=k_new
            graph.inQueue_list[current_node]=True
            graph.expanded_list[current_node]=True
            heapq.heappush(path.queue, (graph.key_list[current_node][0],graph.key_list[current_node][1],current_node))
            
        elif graph.g_list[current_node]>graph.rhs_list[current_node]:
            graph.g_list[current_node]=graph.rhs_list[current_node]
            heapq.heappop(path.queue)
            graph.inQueue_list[current_node]=False

            for p in graph.parents_list[current_node]:
                if p==65535:
                    break

                if p!=path.goal:
                    graph.rhs_list[p]=min(graph.rhs_list[p],graph.g_list[current_node]+compute_c(p,current_node,edges_speed,flow_graph,path.speed,graph))
                update_vertex(path, p,flow_graph,graph)
        else:
            g_old=copy.deepcopy(graph.g_list[current_node])
            graph.g_list[current_node]=float('inf')
            pred_node=current_node

                    
            for p in graph.parents_list[current_node]:
                if p==65535:
                    break
                if graph.rhs_list[p]==(g_old+compute_c(p,current_node,edges_speed,flow_graph,path.speed,graph)):
                    if(p!=path.goal):
                        tt=[]
                        for ch in graph.children_list[p]:
                            if ch==65535:
                                break

                            tt.append(graph.g_list[ch]+compute_c(p,ch,edges_speed,flow_graph,path.speed,graph))
                        graph.rhs_list[p]=min(tt)
                update_vertex(path, p,flow_graph,graph)
            if graph.rhs_list[pred_node]==g_old:
                if pred_node!= path.goal:
                    tt=[]
                    for ch in graph.children_list[pred_node]:
                        if ch==65535:
                            break
                        tt.append(graph.g_list[ch]+compute_c(pred_node,ch,edges_speed,flow_graph,path.speed,graph))
                    graph.rhs_list[pred_node]=min(tt)
            update_vertex(path, pred_node,flow_graph,graph)               

        k_old=top_key(path.queue)
        graph.key_list[path.start]=calculateKey(path.start, path.start, path,flow_graph,graph)
        

    graph.g_list[path.start]=graph.rhs_list[path.start]
            
    return 'Path found!'



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

def angle3pt(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
        Returns a float between 0.0 and 360.0"""
    ang = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

class SearchGraph:
    def __init__(self,key_indices_list,groups_list,parents_list,children_list,g_list,rhs_list,key_list,inQueue_list,expanded_list):
        self.key_indices_list=np.array(key_indices_list,dtype=np.uint16)
        self.groups_list=np.array(groups_list,dtype=np.uint16)

        self.g_list=np.array(g_list,dtype=np.float32) ##TODO: Make sure that uint16 here does not create problems
        self.rhs_list=np.array(rhs_list,dtype=np.float32)
        self.key_list=np.array(key_list,dtype=np.float64)
        self.inQueue_list=np.array(inQueue_list,dtype=np.bool8)
        self.expanded_list=np.array(expanded_list,dtype=np.bool8)

        self.parents_list = np.ones([len(parents_list),len(max(parents_list,key = lambda x: len(x)))],dtype=np.uint16)*65535
        for i,j in enumerate(parents_list):
            self.parents_list[i][0:len(j)] =j
            
        #65535 means empty
        self.children_list = np.ones([len(children_list),len(max(children_list,key = lambda x: len(x)))],dtype=np.uint16)*65535
        for i,j in enumerate(children_list):
            self.children_list[i][0:len(j)] =j

        
class PathPlanning:
    
    def __init__(self,aircraft_type,open_airspace_grid,flow_control_graph,gdf,lon_start,lat_start,lon_dest,lat_dest,exp_const=0.01):
        self.aircraft_type=aircraft_type
        self.start_index=None
        self.start_index_previous=None
        self.start_in_open=True
        self.goal_index=None
        self.goal_index_next=None
        self.dest_in_open=True
        self.open_airspace_grid=open_airspace_grid
        self.flow_graph=flow_control_graph # that is shared memory so it shoudl be fine
        self.flow_control_graph=copy.deepcopy(flow_control_graph)
        self.gdf=gdf
        self.G = None
        self.edge_gdf={} # TODO do that is a smarter way
        self.path=None

        
        self.route=[]
        self.turns=[]
        self.priority=1 #4,3,2,1 in decreasing priority
        self.loitering=False
        self.in_same_cell=False
        self.init_succesful=True
        self.loitering_edges=None

        self.start_point=Point(tuple((lon_start,lat_start)))
        self.goal_point=Point(tuple((lon_dest,lat_dest)))
        self.cutoff_angle=25
        
        self.open_airspace_cells=[]
        
        if self.aircraft_type==1:
            self.speed_max=10.29 ##20 knots
        elif self.aircraft_type==2:
            self.speed_max=15.43 # 30 knots
            
        for i in range(len(self.open_airspace_grid.grid)):
            p=copy.deepcopy(self.open_airspace_grid.grid[i])
   
            y = np.array([[p.p0[0], p.p0[1]], [p.p1[0], p.p1[1]], [p.p2[0] ,p.p2[1]], [p.p3[0], p.p3[1]]])
            self.open_airspace_cells.append([y,i])


        if self.start_in_open:
            point=(lat_start,lon_start)
            geometry, u, v,distance=get_nearest_edge(self.gdf, point)
            if distance<0.000002:
                self.start_index=v
                self.start_index_previous=u
                self.start_in_open=False
            else:
                
                transformer = Transformer.from_crs('epsg:4326','epsg:32633')
                p=transformer.transform(lat_start,lon_start)
                min_x_ind=sort_cells_x(self.open_airspace_cells)
                winner=check_where(p,min_x_ind,self.open_airspace_cells)
                if winner==-1:
                    self.start_in_open=False
                    #point=(lat_start,lon_start)
                    #geometry, u, v,distance=get_nearest_edge(self.gdf, point)
                    self.start_index=v
                    self.start_index_previous=u
                else:
                    open_start=winner[1]
                    self.start_index=self.open_airspace_grid.grid[open_start].key_index
                    self.start_index_previous=5000


        
        if self.dest_in_open:
            point=(lat_dest,lon_dest)
            geometry, u, v,distance=get_nearest_edge(self.gdf, point)
            if distance<0.000002:
                self.goal_index=u
                self.goal_index_next=v
                self.dest_in_open=False
            else:
                transformer = Transformer.from_crs('epsg:4326','epsg:32633')
                p=transformer.transform(lat_dest,lon_dest)
                min_x_ind=sort_cells_x(self.open_airspace_cells)
                winner=check_where(p,min_x_ind,self.open_airspace_cells)
                if winner==-1:
                    self.dest_in_open=False
                    #point=(lat_dest,lon_dest)
                    #geometry, u, v,distance=get_nearest_edge(self.gdf, point)
                    self.goal_index=u
                    self.goal_index_next=v
                else:
                    open_goal=winner[1]
                    self.goal_index=self.open_airspace_grid.grid[open_goal].key_index
                    self.goal_index_next=5000

        del self.open_airspace_cells
        
        
            
        if self.goal_index_next==self.start_index and self.goal_index==self.start_index_previous:
            print("same goal to start index")
            self.init_succesful=False
            return 
        if self.goal_index_next==5000 and self.start_index_previous==5000 and self.start_index==self.goal_index:
            #Start and destination in the same cell
            print("same cell")
            self.in_same_cell=True
            return 
            
            
        #find the area of interest based on teh start and goal point
        ##TODO: tune the exp_const
        if not self.start_in_open and not self.dest_in_open:
            #exp_const=0.01#0.005#0.02##0.005 
            lats=[lat_start,lat_dest,self.flow_graph.nodes_graph[self.start_index].lat,self.flow_graph.nodes_graph[self.start_index_previous].lat,self.flow_graph.nodes_graph[self.goal_index].lat,self.flow_graph.nodes_graph[self.goal_index_next].lat]
            lons=[lon_start,lon_dest,self.flow_graph.nodes_graph[self.start_index].lon,self.flow_graph.nodes_graph[self.goal_index].lon]
            box=bbox(min(lats)-exp_const,min(lons)-exp_const,max(lats)+exp_const,max(lons)+exp_const) 
            
            G,edges=self.flow_control_graph.extract_subgraph(box)
            self.G=copy.deepcopy(G)
            for k in list(edges.keys()):
                for kk in list(edges[k].keys()):
                    key=str(k)+'-'+str(kk)
                    self.edge_gdf[key]=edges[k][kk].speed

        else:
            print('open')
            #exp_const=0.01#0.05#0.03##0.005 
            lats=[lat_start,lat_dest,self.flow_graph.nodes_graph[self.start_index].lat,self.flow_graph.nodes_graph[self.goal_index].lat]
            lons=[lon_start,lon_dest,self.flow_graph.nodes_graph[self.start_index].lon,self.flow_graph.nodes_graph[self.goal_index].lon]
            if self.start_in_open:
                for n in self.flow_graph.nodes_graph[self.start_index].parents:
                    lats.append(self.flow_graph.nodes_graph[n].lat)
                    lons.append(self.flow_graph.nodes_graph[n].lon)
                for n in self.flow_graph.nodes_graph[self.start_index].children:
                    lats.append(self.flow_graph.nodes_graph[n].lat)
                    lons.append(self.flow_graph.nodes_graph[n].lon)
                for n in self.flow_graph.nodes_graph[self.start_index].cell.entry_list:
                    lats.append(self.flow_graph.nodes_graph[n].lat)
                    lons.append(self.flow_graph.nodes_graph[n].lon)    
                for n in self.flow_graph.nodes_graph[self.start_index].cell.exit_list:
                    lats.append(self.flow_graph.nodes_graph[n].lat)
                    lons.append(self.flow_graph.nodes_graph[n].lon)  
            else:
                lats.append(self.flow_graph.nodes_graph[self.start_index_previous].lat)
                lons.append(self.flow_graph.nodes_graph[self.start_index_previous].lon)
            if self.dest_in_open:
                for n in self.flow_graph.nodes_graph[self.goal_index].parents:
                    lats.append(self.flow_graph.nodes_graph[n].lat)
                    lons.append(self.flow_graph.nodes_graph[n].lon)
                for n in self.flow_graph.nodes_graph[self.goal_index].children:
                    lats.append(self.flow_graph.nodes_graph[n].lat)
                    lons.append(self.flow_graph.nodes_graph[n].lon)
                for n in self.flow_graph.nodes_graph[self.goal_index].cell.entry_list:
                    lats.append(self.flow_graph.nodes_graph[n].lat)
                    lons.append(self.flow_graph.nodes_graph[n].lon)    
                for n in self.flow_graph.nodes_graph[self.goal_index].cell.exit_list:
                    lats.append(self.flow_graph.nodes_graph[n].lat)
                    lons.append(self.flow_graph.nodes_graph[n].lon)  
            else:
                lats.append(self.flow_graph.nodes_graph[self.goal_index_next].lat)
                lons.append(self.flow_graph.nodes_graph[self.goal_index_next].lon)                
            box=bbox(min(lats)-exp_const,min(lons)-exp_const,max(lats)+exp_const,max(lons)+exp_const) 
          
            G,edges=self.flow_control_graph.extract_subgraph(box)
            self.G=copy.deepcopy(G)
            for k in list(edges.keys()):
                for kk in list(edges[k].keys()):
                    key=str(k)+'-'+str(kk)
                    self.edge_gdf[key]=edges[k][kk].speed
            
        
        del self.flow_control_graph #empty these, we do not need it any more
        del self.gdf

        #Create the graph
        #self.graph=[]
        
        key_indices_list=[]
        groups_list=[]
        parents_list=[]
        children_list=[]
        g_list=[]
        rhs_list=[]
        key_list=[]
        inQueue_list=[]
        expanded_list=[]
        
        omsnx_keys_list=list(self.G.keys())
        os_keys2_indices=[]
        tmp_dict={}
        tmp_cnt=0
        omsnx_keys_list.sort()
        
        new_nodes_counter=0
        for i in range(len(omsnx_keys_list)):
           key=omsnx_keys_list[i]

           #lon=self.G[key].lon
           #lat=self.G[key].lat 

           parents=self.G[key].parents
           children=self.G[key].children
           my_group={} 
           if self.G[key].open_airspace:
               group=2000 # TO DO lookto do it positiive and unsigned
               #self.os_keys_dict_pred[str(key)+'-0']=i+new_nodes_counter
               
               if key not in tmp_dict.keys():
                   tmp_dict[key]=tmp_cnt
                   tmp_cnt=tmp_cnt+1
                   os_keys2_indices.append([key,i+new_nodes_counter])
               else:
                   os_keys2_indices[tmp_dict[key]].append(i+new_nodes_counter)
               #node=Node(key,np.uint16(i+new_nodes_counter),group)
               key_indices_list.append(key)
               groups_list.append(group)
               g_list.append(float("inf"))
               rhs_list.append(float("inf"))
               key_list.append([0.0,0.0])
               inQueue_list.append(False)
               expanded_list.append(False)
               children_list.append([])
               parents_list.append([])
               my_group.update({i+new_nodes_counter:group})

               #self.graph.append(node)
               continue

        
           ii=0
           tmp=[]#list if the groups that the node has been added
           for p in parents:

               if not ii:
                   if (str(p)+'-'+str(key)) in list(self.edge_gdf.keys()): 

                       group=np.uint16(int(self.flow_graph.edges_graph[p][key].stroke_group))
                       key_indices_list.append(key)
                       groups_list.append(group)
                       g_list.append(float("inf"))
                       rhs_list.append(float("inf"))
                       key_list.append([0.0,0.0])
                       inQueue_list.append(False)
                       expanded_list.append(False)
                       children_list.append([])
                       parents_list.append([])
                      # node=Node(key,np.uint16(i+new_nodes_counter),group)
                       my_group.update({i+new_nodes_counter:group})
                       #self.graph.append(node)
                       tmp.append(group)
                       ii=ii+1
                       #self.os_keys_dict_pred[str(key)+'-'+str(p)]=i+new_nodes_counter

                       if key not in tmp_dict.keys():
                           tmp_dict[key]=tmp_cnt
                           tmp_cnt=tmp_cnt+1
                           os_keys2_indices.append([key,i+new_nodes_counter])

                       elif (i+new_nodes_counter)  not in os_keys2_indices[tmp_dict[key]][1:]:
                           os_keys2_indices[tmp_dict[key]].append(i+new_nodes_counter)

                               
               else: 
                if (str(p)+'-'+str(key)) in list(self.edge_gdf.keys()):  

                        new_nodes_counter=new_nodes_counter+1
                        group=np.uint16(int(self.flow_graph.edges_graph[p][key].stroke_group))
                        key_indices_list.append(key)
                        groups_list.append(group)
                        g_list.append(float("inf"))
                        rhs_list.append(float("inf"))
                        key_list.append([0.0,0.0])
                        inQueue_list.append(False)
                        expanded_list.append(False)
                        children_list.append([])
                        parents_list.append([])
                        #node=Node(key,np.uint16(i+new_nodes_counter),group)
                        my_group.update({i+new_nodes_counter:group})
                        #self.graph.append(node)
                        tmp.append(group)
                        ii=ii+1
                        #self.os_keys_dict_pred[str(key)+'-'+str(p)]=i+new_nodes_counter
                        if key not in tmp_dict.keys():
                           tmp_dict[key]=tmp_cnt
                           tmp_cnt=tmp_cnt+1
                           os_keys2_indices.append([key,i+new_nodes_counter])
 
                               
                        elif (i+new_nodes_counter)  not in os_keys2_indices[tmp_dict[key]][1:]:
                           os_keys2_indices[tmp_dict[key]].append(i+new_nodes_counter)

                           
           for ch in children:
                group=np.uint16(int(self.flow_graph.edges_graph[key][ch].stroke_group))
                if not group in tmp:
                    if not ii:
                        #node=Node(key,np.uint16(i+new_nodes_counter),group)
                        my_group.update({i+new_nodes_counter:group})
                        key_indices_list.append(key)
                        groups_list.append(group)
                        g_list.append(float("inf"))
                        rhs_list.append(float("inf"))
                        key_list.append([0.0,0.0])
                        inQueue_list.append(False)
                        expanded_list.append(False)
                        children_list.append([])
                        parents_list.append([])
                        #self.graph.append(node)
                        tmp.append(group)
                        ii=ii+1
                        if key not in tmp_dict.keys():
                           tmp_dict[key]=tmp_cnt
                           tmp_cnt=tmp_cnt+1
                           os_keys2_indices.append([key,i+new_nodes_counter])
                        elif (i+new_nodes_counter)  not in os_keys2_indices[tmp_dict[key]][1:]:
                            os_keys2_indices[tmp_dict[key]].append(i+new_nodes_counter)

                        
                    else:
                        new_nodes_counter=new_nodes_counter+1
                        #node=Node(key,np.uint16(i+new_nodes_counter),group)
                        key_indices_list.append(key)
                        groups_list.append(group)
                        g_list.append(float("inf"))
                        rhs_list.append(float("inf"))
                        key_list.append([0.0,0.0])
                        inQueue_list.append(False)
                        expanded_list.append(False)
                        children_list.append([])
                        parents_list.append([])
                        my_group.update({i+new_nodes_counter:group})
                        #self.graph.append(node)
                        tmp.append(group)
                        ii=ii+1
                        if key not in tmp_dict.keys():
                           tmp_dict[key]=tmp_cnt
                           tmp_cnt=tmp_cnt+1
                           os_keys2_indices.append([key,i+new_nodes_counter])
                        elif (i+new_nodes_counter)  not in os_keys2_indices[tmp_dict[key]][1:]:
                            os_keys2_indices[tmp_dict[key]].append(i+new_nodes_counter)
                        
           if ii==0:
               #continue
               key_indices_list.append(key)
               groups_list.append(2000)
               g_list.append(float("inf"))
               rhs_list.append(float("inf"))
               key_list.append([0.0,0.0])
               inQueue_list.append(False)
               expanded_list.append(False)
               children_list.append([])
               parents_list.append([])
                #node=Node(key,np.uint16(i+new_nodes_counter),2000)
                #self.graph.append(node)


                #print("No succ or pred: "+str(key))

                        
           if len(my_group)>1:
               for index in my_group:
                   for index_ in my_group:
                        if my_group[index]!=my_group[index_] and index!=index_:
                            children_list[index].append(index_)
                            parents_list[index].append(index_)     
                            
        #add the children and parents to each node                
        for ii,i in enumerate(key_indices_list):
            if self.flow_graph.nodes_graph[i].open_airspace:
                cell=self.flow_graph.nodes_graph[i].cell
                for ch in cell.entry_list:
                    for jj,j in enumerate(key_indices_list):
                        if ch==j:
                            children_list[ii].append(jj)
                            parents_list[jj].append(ii)

                            #self.os_keys_dict_pred[str(j)+'-'+str(i)]=jj
                            if j not in tmp_dict.keys():
                               tmp_dict[j]=tmp_cnt
                               tmp_cnt=tmp_cnt+1
                               os_keys2_indices.append([j,jj])
                            elif jj  not in os_keys2_indices[tmp_dict[j]][1:]:
                               os_keys2_indices[tmp_dict[j]].append(jj)
                   
                            break                    
                for p in cell.exit_list:
                    for jj,j in enumerate(key_indices_list):
                        if p==j :
                            parents_list[ii].append(jj)
                            children_list[jj].append(ii)

                            #self.os_keys_dict_pred[str(i)+'-'+str(p)]=ii
                            if i not in tmp_dict.keys():
                               tmp_dict[i]=tmp_cnt
                               tmp_cnt=tmp_cnt+1
                               os_keys2_indices.append([i,ii])
                            elif ii  not in os_keys2_indices[tmp_dict[i]][1:]:
                               os_keys2_indices[tmp_dict[i]].append(ii)
                            break
                        
                for p in self.flow_graph.nodes_graph[i].parents:
                    if self.flow_graph.nodes_graph[p].open_airspace:
                        for jj,j in enumerate(key_indices_list):
                            if p==j:
                                #print(p)
                                parents_list[ii].append(jj)
                                children_list[ii].append(jj)
                                break
            else:
                key=i
                parents=self.G[key].parents
                children=self.G[key].children
                for p in parents:
                    for jj,j in enumerate(key_indices_list):
                        if p==j and (groups_list[jj]==groups_list[ii] or groups_list[ii] ==2000) :
                            parents_list[ii].append(jj)
                            #self.os_keys_dict_pred[str(i)+'-'+str(p)]=ii                    
                            if i not in tmp_dict.keys():
                               tmp_dict[i]=tmp_cnt
                               tmp_cnt=tmp_cnt+1
                               os_keys2_indices.append([i,ii])
                            elif ii  not in os_keys2_indices[tmp_dict[i]][1:]:
                               os_keys2_indices[tmp_dict[i]].append(ii)                            
                            break
              
                for ch in children:
                    for jj,j in enumerate(key_indices_list):
                        if ch==j and (groups_list[jj]==groups_list[ii] or groups_list[ii] ==2000) :
                            children_list[ii].append(jj)
                            if i not in tmp_dict.keys():
                               tmp_dict[i]=tmp_cnt
                               tmp_cnt=tmp_cnt+1
                               os_keys2_indices.append([i,ii])
                            elif ii  not in os_keys2_indices[tmp_dict[i]][1:]:
                               os_keys2_indices[tmp_dict[i]].append(ii)                      
                            break
          

        

        del self.open_airspace_grid
        del self.G
        del self.edge_gdf
        self.graph=SearchGraph(key_indices_list,groups_list,parents_list,children_list,g_list,rhs_list,key_list,inQueue_list,expanded_list)
        self.os_keys2_indices = np.ones([len(os_keys2_indices),len(max(os_keys2_indices,key = lambda x: len(x)))],dtype=np.uint16)*65535
        for i,j in enumerate(os_keys2_indices):
            self.os_keys2_indices[i][0:len(j)] =j
  
        

        

    ##Function handling the path planning process
    ##Retruns: route,turns,edges_list,next_turn_point,groups,in_constrained,turn_speed
    ##route is the list of waypoints (lon,lat)
    ##turns is the list of booleans indicating for every waypoint if it is a turn
    ##edges_list is the list of the edges
    ##next_turn_point is a list containing the coord of every point that is a turn point
    ##groups is the list of the group in which each waypoint belongs to
    ##in_constrained is the list of booleans indicating for every waypoint if it is in constarined airspace
    ##turn_speed is teh list if speed to be used if the waypoint is a turning waypoint   
    def plan(self):

        
        if self.in_same_cell:
            self.route=[(self.start_point.x,self.start_point.y),(self.goal_point.x,self.goal_point.y)]
            self.turns=[False,True]
            self.edges_list=[(self.start_index_previous,self.start_index),(self.goal_index_next,self.goal_index)]
            self.next_turn_point=[(-999,-999),(-999,-999)]
            self.groups=[2000,2000]    
            self.in_constrained=[False,False]
            self.turn_speed=[0,5]
            
            return self.route,self.turns,self.edges_list,self.next_turn_point,self.groups,self.in_constrained,self.turn_speed


# =============================================================================
#         start_id=self.os_keys_dict_pred[str(self.start_index)+'-'+str(self.start_index_previous)]
#         print(self.goal_index,self.goal_index_next)
#         if self.goal_index_next==0:
#             goal_id=self.os_keys_dict_pred[str(self.goal_index)+'-'+str(0)] 
#         else:
#             goal_list=self.graph.parents_list[self.os_keys_dict_pred[str(self.goal_index_next)+'-'+str(self.goal_index)] ]
#             for p in goal_list:
#                 if p==65535:
#                     break
#                 if self.graph.key_indices_list[p]==self.goal_index:
#                     
#                     goal_id=p
#                     break
# =============================================================================
           
        start_id=None
        goal_id=None
        
        ##########
        
        result = np.where(self.os_keys2_indices ==self.start_index)
        rr=np.where(result[1] ==0)
        if self.start_index_previous==5000:
            start_id=self.os_keys2_indices[result[0][rr]][0][1]
        else:
            for ii in self.os_keys2_indices[result[0][rr]][0][1:]:
                if ii==65535:
                    break
                for p in self.graph.parents_list[ii]:
                    if p ==65535:
                        break
                    if self.start_index_previous==self.graph.key_indices_list[p]:
                        start_id=ii
                        break
                if start_id !=None:
                    break
        result = np.where(self.os_keys2_indices ==self.goal_index)
        rr=np.where(result[1] ==0)
        if self.goal_index_next==5000:
            goal_id=self.os_keys2_indices[result[0][rr]][0][1]
        else:
            for ii in self.os_keys2_indices[result[0][rr]][0][1:]:
                if ii==65535:
                    break
                for p in self.graph.children_list[ii]:
                    if p ==65535:
                        break
                    if self.goal_index_next==self.graph.key_indices_list[p]:
                        goal_id=ii
                        break
                if goal_id !=None:
                    break 
        #########
        
# =============================================================================
#         for i in self.os_keys2_indices:
#             key=i[0]
#             if key==self.start_index:
#                 if self.start_index_previous==0:
#                     start_id=i[1]
#                 else:
#                     for ii in i[1:]:
#                         if ii==65535:
#                             break
#                         for p in self.graph.parents_list[ii]:
#                             if p ==65535:
#                                 break
#                             if self.start_index_previous==self.graph.key_indices_list[p]:
#                                 start_id=ii
#                                 break
#                         if start_id !=None:
#                             break
#             if key==self.goal_index:
#                 if self.goal_index_next==0:
#                     goal_id=i[1]  
#                 else:
#                     for ii in i[1:]:
#                         if ii==65535:
#                             break
#                         for ch in self.graph.children_list[ii]:
#                             if ch==65535:
#                                 break
#                             if self.goal_index_next==self.graph.key_indices_list[ch]:
#                                 goal_id=ii  
#                                 break
#                         if goal_id !=None:
#                             break                        
#             if start_id !=None and goal_id !=None:
#                 break
# =============================================================================
            
        start_node=start_id
        goal_node=goal_id


        self.path=Path(start_node,goal_node,self.speed_max,self.graph)
        
        initialise(self.path,self.flow_graph)
        
        path_found=compute_shortest_path(self.path,self.graph,self.flow_graph.edges_init_speed,self.flow_graph)
        print(path_found)
        
        route=[]
        turns=[]
        edges_list=[]
        next_turn_point=[]
        indices_nodes=[]
        #turn_indices=[]
        if path_found:
            route,turns,indices_nodes,turn_coord,groups,in_constrained,turn_speed,init_groups=self.get_path(self.path,self.graph,self.flow_graph.edges_init_speed,self.flow_graph.edges_graph)

            if route==None:
                #No path was found
                return [],[],[],[],[],[],[]
            
            os_id1=self.start_index_previous

            os_id2=indices_nodes[0]
                
            if 2000 not in init_groups and self.start_index_previous==5000:
                    edges_list.append((os_id1,os_id2))

                    nodes_index=0

                    for i in range(len(init_groups)-1):
                        edges_list.append((os_id1,os_id2))

                        if nodes_index>len(init_groups)-2:
                            break
                        if nodes_index<len(init_groups)-1 and indices_nodes[nodes_index+1]==os_id2:
                            nodes_index=nodes_index+1
                            continue
    
                        nodes_index=nodes_index+1
                        os_id1=os_id2
                        os_id2=indices_nodes[nodes_index]
                
            else:
                    if not( init_groups[0]==2000 and init_groups[2]!=2000) :
                        edges_list.append((os_id1,os_id2))
                    
                    nodes_index=1
    
                    for i in range(len(init_groups)-1):
                        edges_list.append((os_id1,os_id2))
                        if nodes_index>len(init_groups)-2:
                            break
                        if nodes_index<len(init_groups)-1 and indices_nodes[nodes_index+1]==os_id2:
                            nodes_index=nodes_index+1
                            continue
                        if nodes_index<len(init_groups)-2 and init_groups[nodes_index+1]==2000 and init_groups[nodes_index+2]!=2000:
                            nodes_index=nodes_index+1
                            os_id2=indices_nodes[nodes_index]
    
                        if init_groups[nodes_index]==2000 and init_groups[nodes_index+1]==2000:
                            nodes_index=nodes_index+1
                            os_id1=5000
                            os_id2=indices_nodes[nodes_index]
    
                        else:
                            nodes_index=nodes_index+1
                            os_id1=os_id2
                            os_id2=indices_nodes[nodes_index]


                
            cnt=0
            for i in range(len(turns)):
                if turns[i]:# and in_constrained[i]:
                    next_turn_point.append(turn_coord[cnt])
                    cnt=cnt+1
                else:
                    next_turn_point.append(turn_coord[cnt])
                    
        del indices_nodes[0]
                    
        turns[-1]=True
        turn_speed[-1]=5
        
        self.route=np.array(route,dtype=np.float64)
        self.turns=np.array(turns,dtype=np.bool8) 
        self.edges_list=np.array(edges_list) 
        self.next_turn_point=next_turn_point
        self.groups=np.array(groups,dtype=np.uint16)
        self.in_constrained=np.array(in_constrained,dtype=np.bool8)
        self.turn_speed=np.array(turn_speed,dtype=np.uint16())
        
        return route,turns,edges_list,next_turn_point,groups,in_constrained,turn_speed
    
    ##Function to export the route based on the D* search graph
    ##Retruns: route,turns,next_node_index,turn_coord,groups
    ##route is the list of waypoints (lon,lat)
    ##turns is the list of booleans indicating for every waypoint if it is a turn
    ##next_node_index is the list of the next osmnx node for every waypoint
    ##turn_coord is a list containing the coord of every point that is a turn point
    ##groups_numbers is the list of the group in which each waypoint belongs to
    ##in_constrained is the list of booleans indicating for every waypoint if it is in constarined airspace
    ##turn_speed is teh list if speed to be used if the waypoint is a turning waypoint
    def get_path(self,path,graph,edges_speed, edges,edges_old=None,change=False,change_list=[]):

        route_centers=[]
        next_node_index=[]
        group_numbers=[]
        turns=[]
        turn_coords=[]
        airspace_transitions=[]
        in_open_airspace=False
        in_constrained=[] #list indicating in every waypoint is in constrained airspace (value=1) or in open airspace (value =0)
        
        path_found=True
        
        if change: #Scan for changes
        ##replan
            path.k_m=path.k_m+heuristic(path.origin_node_index,self.path.start,path.speed,self.flow_graph,graph)
            for c in change_list:
                
                #if not  graph.expanded_list[c[0]] or not graph.expanded_list[c[1]]:
                   # print("not expanded")
                    

                c_old=compute_c(c[0], c[1],edges_old,self.flow_graph,path.speed,graph)

                    #update cost and obstacles here
                if c_old>compute_c(c[0], c[1],edges_speed,self.flow_graph,path.speed,graph): #if cost is decreased

                    if(c[0]!=path.goal):
    
                        graph.rhs_list[c[0]]=min(graph.rhs_list[c[0]],compute_c(c[0], c[1], edges_speed,self.flow_graph,path.speed,graph)+graph.g_list[c[1]])

                            
                elif graph.rhs_list[c[0]]== c_old+graph.g_list[c[1]]: #if cost is increased

                    if c[0]!=path.goal:
                        tt=[]
                        for ch in graph.children_list[c[0]]:
                            if ch==65535:
                                break
                            tt.append(graph.g_list[ch]+compute_c( c[0],ch, edges_speed,self.flow_graph,path.speed,graph))
                        graph.rhs_list[c[0]]=min(tt)
                        graph.rhs_list[path.start]=float('inf')## not sure for that

                update_vertex(path, c[0],self.flow_graph,graph)
                
                edges_old[graph.key_indices_list[c[0]]][graph.key_indices_list[c[1]]]=edges_speed[graph.key_indices_list[c[0]]][graph.key_indices_list[c[1]]]
            path_found=compute_shortest_path(path,graph,edges_old,self.flow_graph)


            print(path_found)
            change=False 
            if not  path_found:
                print("Compute path all from the start")
                #break
                
        if not  path_found:
            length=len(graph.g_list)
            graph.g_list=np.ones(length,dtype=np.float32)*float('inf')
            graph.rhs_list=np.ones(length,dtype=np.float32)*float('inf')
            graph.inQueue_list=np.ones(length,dtype=np.bool8)*False
            graph.expanded_list=np.ones(length,dtype=np.bool8)*False
            graph.key_list=np.zeros([length,2],dtype=np.float64)
            self.route_origin_node=copy.deepcopy(path.start)
                    
            initialise(self.path,self.flow_graph)
            path_found=compute_shortest_path(path,graph,edges_speed,self.flow_graph) 
            
        if not path_found:
            return None,None,None,None,None,None,None,None
        

        next_node_index.append(self.start_index)
        tmp=(self.start_point.x,self.start_point.y)
        group_numbers.append(graph.groups_list[path.start])
        route_centers.append(tmp)
        turns.append(False)
        
        if self.flow_graph.nodes_graph[graph.key_indices_list[path.start]].open_airspace:
            next_node_index.append(self.start_index)
            tmp=(self.start_point.x,self.start_point.y)
            group_numbers.append(graph.groups_list[path.start])
            turns.append(False)

        if not self.flow_graph.nodes_graph[graph.key_indices_list[path.start]].open_airspace and not self.flow_graph.nodes_graph[self.start_index_previous].open_airspace :    

            linestring=edges[self.start_index_previous][self.start_index].geometry
            coords = list(linestring.coords)
            for c in range(len(coords)-1):
                if (not c==0) and (lies_between(tuple((coords[c][0],coords[c][1])),tuple((self.start_point.x,self.start_point.y)),tuple((self.flow_graph.nodes_graph[graph.key_indices_list[path.start]].lon,self.flow_graph.nodes_graph[graph.key_indices_list[path.start]].lat)))):
                    tmp=(coords[c][0],coords[c][1]) #the points before the first node
                    route_centers.append(tmp) 
                    group_numbers.append(graph.groups_list[path.start])
                    next_node_index.append(self.start_index)
                    turns.append(False)
                

            next_node_index.append(self.start_index)
            
            tmp=(self.flow_graph.nodes_graph[graph.key_indices_list[path.start]].lon,self.flow_graph.nodes_graph[graph.key_indices_list[path.start]].lat)
            group_numbers.append(graph.groups_list[path.start])
            route_centers.append(tmp)
            turns.append(False)
        if graph.groups_list[path.start] ==2000:
            airspace_transitions.append(0)
            in_open_airspace=True
        
        group=graph.groups_list[path.start]           

        
        selected_nodes_index=[]
        selected_nodes_index.append(path.start)


        while graph.key_indices_list[path.start]!=graph.key_indices_list[path.goal] :
            
    
            current_node=path.start
            minim=float('inf')
  
            for ch in graph.children_list[path.start]:
                if ch==65535:
                    break
               
                if compute_c(path.start, ch,edges_speed,self.flow_graph,path.speed,graph)+graph.g_list[ch]<minim:
                    minim=compute_c(path.start, ch,edges_speed,self.flow_graph,path.speed,graph)+graph.g_list[ch]
                    current_node=ch
                    
                    
            if current_node in selected_nodes_index:
                print(selected_nodes_index)
                print(current_node)
                print("get_path stack !! Please report this!")
                return None,None,None,None,None,None,None,None
                
            selected_nodes_index.append(current_node)
            

                    
            if graph.key_indices_list[current_node]!=graph.key_indices_list[path.start]:
                
                if not self.flow_graph.nodes_graph[graph.key_indices_list[current_node]].open_airspace and not self.flow_graph.nodes_graph[graph.key_indices_list[path.start]].open_airspace:
                    #find the intermediate points
                    #pp=1
                    linestring=edges[graph.key_indices_list[path.start]][graph.key_indices_list[current_node]].geometry #if the start index should go first need to get checked
                    coords = list(linestring.coords)
                    for c in range(len(coords)-1):
                        if not c==0:
                            tmp=(coords[c][0],coords[c][1]) #the intermediate point
                            route_centers.append(tmp) 
                            group_numbers.append(graph.groups_list[current_node])
                            next_node_index.append(graph.key_indices_list[current_node])
                            turns.append(False)

                if graph.key_indices_list[current_node]!=graph.key_indices_list[path.goal] or not self.flow_graph.nodes_graph[graph.key_indices_list[current_node]].open_airspace:

                    next_node_index.append(graph.key_indices_list[current_node])
                    tmp=(self.flow_graph.nodes_graph[graph.key_indices_list[current_node]].lon,self.flow_graph.nodes_graph[graph.key_indices_list[current_node]].lat) #the next node
                    route_centers.append(tmp)
                    
                    group_numbers.append(graph.groups_list[current_node])
                    turns.append(False) 

                    
                    if graph.groups_list[current_node]==2000 and not in_open_airspace:

                        airspace_transitions.append(len(group_numbers)-1)
                        in_open_airspace=True
                    elif graph.groups_list[current_node]!=2000 and  in_open_airspace:

                        airspace_transitions.append(len(group_numbers)-1)
                        in_open_airspace=False
                        tmp=(self.flow_graph.nodes_graph[graph.key_indices_list[current_node]].lon,self.flow_graph.nodes_graph[graph.key_indices_list[current_node]].lat) #the next node
                        route_centers.append(tmp)
                
   
            path.start=current_node
            
           
          
        if not self.flow_graph.nodes_graph[graph.key_indices_list[path.goal]].open_airspace: 
    
            linestring=edges[self.goal_index][self.goal_index_next].geometry
            coords = list(linestring.coords)
            for c in range(len(coords)-1):
                if (not c==0) and (lies_between(tuple((coords[c][0],coords[c][1])),tuple((self.goal_point.x,self.goal_point.y)),tuple((self.flow_graph.nodes_graph[graph.key_indices_list[path.goal]].lon,self.flow_graph.nodes_graph[graph.key_indices_list[path.goal]].lat)))):
                    tmp=(coords[c][0],coords[c][1]) #the points before the first node
                    route_centers.append(tmp) 
                    group_numbers.append(graph.groups_list[path.goal])
                    next_node_index.append(self.goal_index_next)
                    turns.append(False)
                

            tmp=(self.goal_point.x,self.goal_point.y)
            route_centers.append(tmp)
            group_numbers.append(graph.groups_list[path.goal])
            next_node_index.append(self.goal_index_next)
            turns.append(False)   
        else:
            tmp=(self.goal_point.x,self.goal_point.y)
            route_centers.append(tmp)
            group_numbers.append(graph.groups_list[path.goal])
            next_node_index.append(self.goal_index)
            turns.append(False)  

        if in_open_airspace:
            airspace_transitions.append(len(route_centers)-1)
            in_open_airspace=False

        delete_indices=[]
        p3=None

        if len(airspace_transitions)==0:
            route=route_centers
        else:
            open_i=0
            transformer1 = Transformer.from_crs( 'epsg:32633','epsg:4326')
            
            route=[]
            route.append(route_centers[0])

            if airspace_transitions[open_i]>0:
                p1=[route_centers[airspace_transitions[open_i]-1][0],route_centers[airspace_transitions[open_i]-1][1]]
            else:
                p1=[route_centers[airspace_transitions[open_i]][0],route_centers[airspace_transitions[open_i]][1]]
            if airspace_transitions[open_i+1]-airspace_transitions[open_i]>10:

                p2=[route_centers[airspace_transitions[open_i+1]][0],route_centers[airspace_transitions[open_i+1]][1]]
                transformer2 = Transformer.from_crs( 'epsg:4326','epsg:32633')
                
                p1=transformer2.transform(p1[1],p1[0])
                p2=transformer2.transform(p2[1],p2[0])  
                shapely_line = shapely.geometry.LineString([(p1[0],p1[1]),(p2[0],p2[1])])
                intersection_line =self.flow_graph.constrained_poly.intersection(shapely_line)
                if not intersection_line.is_empty:
                    p3=p2

                    p2=[route_centers[int((airspace_transitions[open_i+1]-airspace_transitions[open_i])/2)][0],route_centers[int((airspace_transitions[open_i+1]-airspace_transitions[open_i])/2)][1]]
                    p2=transformer2.transform(p2[1],p2[0])
  
                    shapely_line1 = shapely.geometry.LineString([(p1[0],p1[1]),(p2[0],p2[1])])
                    shapely_line2 = shapely.geometry.LineString([(p3[0],p3[1]),(p2[0],p2[1])])

                    intersection_line1 = self.flow_graph.constrained_poly.intersection(shapely_line1)
                    intersection_line2 = self.flow_graph.constrained_poly.intersection(shapely_line2)
                    if not intersection_line1.is_empty and not intersection_line2.is_empty:
                        a=1
                    elif not intersection_line2.is_empty:
                        cnt=0
                        while intersection_line1.is_empty:
                            cnt=cnt+1
                            p2=[route_centers[cnt+int((airspace_transitions[open_i+1]-airspace_transitions[open_i])/2)][0],route_centers[cnt+int((airspace_transitions[open_i+1]-airspace_transitions[open_i])/2)][1]]
                            p2=transformer2.transform(p2[1],p2[0])
          
                            shapely_line1 = shapely.geometry.LineString([(p1[0],p1[1]),(p2[0],p2[1])])
                            shapely_line2 = shapely.geometry.LineString([(p3[0],p3[1]),(p2[0],p2[1])])
        
                            intersection_line1 = self.flow_graph.constrained_poly.intersection(shapely_line1)
                            intersection_line2 = self.flow_graph.constrained_poly.intersection(shapely_line2) 
                            if intersection_line2.is_empty:
                                cnt=cnt+1
                                break
                        p2=[route_centers[cnt-1+int((airspace_transitions[open_i+1]-airspace_transitions[open_i])/2)][0],route_centers[cnt-1+int((airspace_transitions[open_i+1]-airspace_transitions[open_i])/2)][1]]
                        p2=transformer2.transform(p2[1],p2[0])
                    elif not intersection_line1.is_empty:
                        cnt=0
                        while intersection_line2.is_empty:
                            cnt=cnt-1
                            p2=[route_centers[cnt+int((airspace_transitions[open_i+1]-airspace_transitions[open_i])/2)][0],route_centers[cnt+int((airspace_transitions[open_i+1]-airspace_transitions[open_i])/2)][1]]
                            p2=transformer2.transform(p2[1],p2[0])
          
                            shapely_line1 = shapely.geometry.LineString([(p1[0],p1[1]),(p2[0],p2[1])])
                            shapely_line2 = shapely.geometry.LineString([(p3[0],p3[1]),(p2[0],p2[1])])
        
                            intersection_line1 = self.flow_graph.constrained_poly.intersection(shapely_line1)
                            intersection_line2 = self.flow_graph.constrained_poly.intersection(shapely_line2) 
                            if intersection_line1.is_empty:
                                cnt=cnt-1
                                break
                        p2=[route_centers[cnt+1+int((airspace_transitions[open_i+1]-airspace_transitions[open_i])/2)][0],route_centers[cnt+1+int((airspace_transitions[open_i+1]-airspace_transitions[open_i])/2)][1]]
                        p2=transformer2.transform(p2[1],p2[0])
                        
            else:
                p2=[route_centers[airspace_transitions[open_i+1]][0],route_centers[airspace_transitions[open_i+1]][1]]
                transformer2 = Transformer.from_crs( 'epsg:4326','epsg:32633')
                
                p1=transformer2.transform(p1[1],p1[0])
                p2=transformer2.transform(p2[1],p2[0])  

            
            
            for j in range(len(next_node_index)-1): 
                
                if j+1>airspace_transitions[open_i+1]:
                    if open_i+2<len(airspace_transitions):
                        open_i=open_i+2
                        p1=[route_centers[airspace_transitions[open_i]-1][0],route_centers[airspace_transitions[open_i]-1][1]]
                        transformer2 = Transformer.from_crs( 'epsg:4326','epsg:32633')
                        p1=transformer2.transform(p1[1],p1[0])
                        
                        if airspace_transitions[open_i+1]-airspace_transitions[open_i]>10:

                            p2=[route_centers[airspace_transitions[open_i+1]][0],route_centers[airspace_transitions[open_i+1]][1]]
                            transformer2 = Transformer.from_crs( 'epsg:4326','epsg:32633')
                            
                            p2=transformer2.transform(p2[1],p2[0])  
                            shapely_line = shapely.geometry.LineString([(p1[0],p1[1]),(p2[0],p2[1])])
                            intersection_line =self.flow_graph.constrained_poly.intersection(shapely_line)
                            if not intersection_line.is_empty:
                                p3=p2
            
                                p2=[route_centers[int((airspace_transitions[open_i+1]-airspace_transitions[open_i])/2)][0],route_centers[int((airspace_transitions[open_i+1]-airspace_transitions[open_i])/2)][1]]
                                p2=transformer2.transform(p2[1],p2[0])
              
                                shapely_line1 = shapely.geometry.LineString([(p1[0],p1[1]),(p2[0],p2[1])])
                                shapely_line2 = shapely.geometry.LineString([(p3[0],p3[1]),(p2[0],p2[1])])
            
                                intersection_line1 = self.flow_graph.constrained_poly.intersection(shapely_line1)
                                intersection_line2 = self.flow_graph.constrained_poly.intersection(shapely_line2)
                                if not intersection_line1.is_empty and not intersection_line2.is_empty:
                                    a=1
                                elif not intersection_line2.is_empty:
                                    cnt=0
                                    while intersection_line1.is_empty:
                                        cnt=cnt+1
                                        p2=[route_centers[cnt+int((airspace_transitions[open_i+1]-airspace_transitions[open_i])/2)][0],route_centers[cnt+int((airspace_transitions[open_i+1]-airspace_transitions[open_i])/2)][1]]
                                        p2=transformer2.transform(p2[1],p2[0])
                      
                                        shapely_line1 = shapely.geometry.LineString([(p1[0],p1[1]),(p2[0],p2[1])])
                                        shapely_line2 = shapely.geometry.LineString([(p3[0],p3[1]),(p2[0],p2[1])])
                    
                                        intersection_line1 = self.flow_graph.constrained_poly.intersection(shapely_line1)
                                        intersection_line2 = self.flow_graph.constrained_poly.intersection(shapely_line2) 
                                        if intersection_line2.is_empty:
                                            cnt=cnt+1
                                            break
                                    p2=[route_centers[cnt-1+int((airspace_transitions[open_i+1]-airspace_transitions[open_i])/2)][0],route_centers[cnt-1+int((airspace_transitions[open_i+1]-airspace_transitions[open_i])/2)][1]]
                                    p2=transformer2.transform(p2[1],p2[0])
                                elif not intersection_line1.is_empty:
                                    cnt=0
                                    while intersection_line2.is_empty:
                                        cnt=cnt-1
                                        p2=[route_centers[cnt+int((airspace_transitions[open_i+1]-airspace_transitions[open_i])/2)][0],route_centers[cnt+int((airspace_transitions[open_i+1]-airspace_transitions[open_i])/2)][1]]
                                        p2=transformer2.transform(p2[1],p2[0])
                      
                                        shapely_line1 = shapely.geometry.LineString([(p1[0],p1[1]),(p2[0],p2[1])])
                                        shapely_line2 = shapely.geometry.LineString([(p3[0],p3[1]),(p2[0],p2[1])])
                    
                                        intersection_line1 = self.flow_graph.constrained_poly.intersection(shapely_line1)
                                        intersection_line2 = self.flow_graph.constrained_poly.intersection(shapely_line2) 
                                        if intersection_line1.is_empty:
                                            cnt=cnt-1
                                            break
                                    p2=[route_centers[cnt+1+int((airspace_transitions[open_i+1]-airspace_transitions[open_i])/2)][0],route_centers[cnt+1+int((airspace_transitions[open_i+1]-airspace_transitions[open_i])/2)][1]]
                                    p2=transformer2.transform(p2[1],p2[0])
                            

                        else:
                            p2=[route_centers[airspace_transitions[open_i+1]][0],route_centers[airspace_transitions[open_i+1]][1]]
                            p2=transformer2.transform(p2[1],p2[0])
                            
                if p3!=None:
                    if j>int((airspace_transitions[open_i+1]-airspace_transitions[open_i])/2)+airspace_transitions[open_i]:
                        p1=p2
                        p2=p3
                        p3=None    
                        
                if  group_numbers[j]!=2000 and j!=0: 
                    route.append(route_centers[j])



                else:
                    if j==0 and  group_numbers[j]!=2000:
                        i=self.start_index_previous
                    elif group_numbers[j]!=2000:
                        i=next_node_index[j-1]
                    else:
                        i=0
    
                    ii=next_node_index[j]
                    
                    node1=self.flow_graph.nodes_graph[ii]

                    #node1=self.flow_graph.nodes_graph[graph.key_indices_list[self.os_keys_dict_pred[str(ii)+'-'+str(i)]].key_index]

                    if group_numbers[j]!=2000 or group_numbers[j+1]!=2000:
                        i=next_node_index[j]
                    else:
                        i=0
                    
                    ii=next_node_index[j+1]
                    
                    
                    if group_numbers[j]==2000 and group_numbers[j+1]!=2000:
                        delete_indices.append(j+1)

                    if group_numbers[j+1]!=2000 or next_node_index[j+1]==next_node_index[j]: ##That should be deleted

                        continue
                    


                    #node2=self.flow_graph.nodes_graph[self.graph[self.os_keys_dict_pred[str(ii)+'-'+str(i)]].key_index]
                    node2=self.flow_graph.nodes_graph[ii]
                    
                    if node1.cell.p0[0]==node2.cell.p2[0] :
                        ymin=max(node1.cell.p1[1],node2.cell.p2[1])
                        ymax=min(node1.cell.p0[1],node2.cell.p3[1])
                        edge=[[node1.cell.p0[0],ymin],[node1.cell.p0[0],ymax]]
                    elif node1.cell.p2[0]==node2.cell.p0[0]:
                        ymin=max(node1.cell.p2[1],node2.cell.p1[1])
                        ymax=min(node1.cell.p3[1],node2.cell.p0[1])
                        edge=[[node1.cell.p2[0],ymin],[node1.cell.p2[0],ymax]]



                    pp1=find_closest_point_on_linesegment(edge,p1)

                    intersection=intersect(p1,p2,edge[0],edge[1])
                    if intersection:
                        px,py=line_intersection_point([p1,p2], edge)
                    else:
                        px=pp1[0]
                        py=pp1[1]

                    p=[px,py]                   
                    p1=p
                          
                    tranformed_p=transformer1.transform(px,py)
                    lon,lat =tranformed_p[1],tranformed_p[0]
              
                    tmp=(lon,lat)
                    route.append(tmp)


                    
            route.append(route_centers[len(route_centers)-1])
            
            
        init_groups=copy.deepcopy(group_numbers)
        for i in range(len(delete_indices)):

            
            j=delete_indices[len(delete_indices)-1-i]
            del group_numbers[j]
            del turns[j]


            
        ##Check for turn points
        lat_prev=self.start_point.y
        lon_prev=self.start_point.x
        
        turn_speed=copy.deepcopy(turns) ## TODO : update those values when they are ready
        #speed set to 0 for open airspace or for no turn
        #speed to 10 knots for angles smaller than 45 degrees
        #speed to 5 knots for turning angles between 45 and 90 degrees
        #speed to 2 knots for turning angles larger tha 90 degrees


        for i in range(1,len(group_numbers)-1):
            lat_cur=route[i][1]
            lon_cur=route[i][0]
            lat_next=route[i+1][1]
            lon_next=route[i+1][0]
            ##Check the angle between the prev point- current point and the current point- next point  
            #line_string_1 = [(lat_prev,lon_prev), (lat_cur,lon_cur)]
            #line_string_2 = [(lat_cur,lon_cur), (lat_next,lon_next)]
            d1=qdrdist(lat_prev,lon_prev,lat_cur,lon_cur)
            d2=qdrdist(lat_cur,lon_cur,lat_next,lon_next)

            angle=abs(d2[0]-d1[0])
            
            if angle>180:
                angle=360-angle


            if angle>self.cutoff_angle and group_numbers[i]!=2000 and i!=0:
                turns[i]=True
                tmp=(route[i][1],route[i][0])
                turn_coords.append(tmp)
                if angle<100:
                    turn_speed[i]=10
                elif angle<150:
                    turn_speed[i]=5
                else:
                    turn_speed[i]=2

                
            lat_prev=lat_cur
            lon_prev=lon_cur
            if group_numbers[i]==group_numbers[i+1] or group_numbers[i+1]==2000:
                continue
            elif turns[i]!=True:
                turns[i]=True
                tmp=(route[i][1],route[i][0])
                turn_coords.append(tmp)
                if not turn_speed[i]:
                    turn_speed[i]=10

        turn_coords.append((-999,-999))
        turns[0]=False

        for g,i in enumerate(group_numbers):
            if i==2000 :#or ( g>0 and group_numbers[g-1]==-1)or ( g<len(group_numbers)-1 and group_numbers[g+1]==-1):
                in_constrained.append(False)
            else:
                in_constrained.append(True)        



        return route,turns,next_node_index,turn_coords,group_numbers,in_constrained,turn_speed,init_groups
 
    def update_changed_vertices(self,path,graph,edges_speed,edges,edges_old,change=True,change_list=[]):
        
        if change: #Scan for changes
        ##replan

            path.k_m=path.k_m+heuristic(path.origin_node_index,path.start,path.speed,self.flow_graph,graph)
            for c in change_list:
                
                #if not  graph.expanded_list[c[0]] or not graph.expanded_list[c[1]]:
                    #print("not expanded")
                    

                c_old=compute_c(c[0], c[1],edges_old,self.flow_graph,path.speed,graph)

                    #update cost and obstacles here
                if c_old>compute_c(c[0], c[1],edges_speed,self.flow_graph,path.speed,graph): #if cost is decreased

                    if(c[0]!=path.goal):
    
                        graph.rhs_list[c[0]]=min(graph.rhs_list[c[0]],compute_c(c[0], c[1], edges_speed,self.flow_graph,path.speed,graph)+graph.g_list[c[1]])

                            
                elif graph.rhs_list[c[0]]== c_old+graph.g_list[c[1]]: #if cost is increased

                    if c[0]!=path.goal:
                        tt=[]
                        for ch in graph.children_list[c[0]]:
                            if ch==65535:
                                break
                            tt.append(graph.g_list[ch]+compute_c( c[0],ch, edges_speed,self.flow_graph,path.speed,graph))
                        graph.rhs_list[c[0]]=min(tt)
                        graph.rhs_list[path.start]=float('inf')## not sure for that

                update_vertex(path, c[0],self.flow_graph,graph)
                
                edges_old[graph.key_indices_list[c[0]]][graph.key_indices_list[c[1]]]=edges_speed[graph.key_indices_list[c[0]]][graph.key_indices_list[c[1]]]

 
    ##Function handling the replanning process, called when flow control is updated
    ##Returns: route,turns,edges_list,next_turn_point,groups,in_constrained,turn_speed
    ##route is the list of waypoints (lon,lat)
    ##turns is the list of booleans indicating for every waypoint if it is a turn
    ##edges_list is the list of the edges
    ##next_turn_point is a list containing the coord of every point that is a turn point
    ##groups is the list of the group in which each waypoint belongs to
    ##in_constrained is the list of booleans indicating for every waypoint if it is in constarined airspace
    ##turn_speed is teh list if speed to be used if the waypoint is a turning waypoint        
    def replan(self,changes_list,prev_node_osmnx_id,next_node_index,lat,lon):
        if self.in_same_cell:
            self.route=[(self.start_point.x,self.start_point.y),(self.goal_point.x,self.goal_point.y)]
            self.turns=[False,True]
            self.edges_list=[(self.start_index_previous,self.start_index),(self.goal_index_next,self.goal_index)]
            self.next_turn_point=[(-999,-999),(-999,-999)]
            self.groups=[2000,2000]    
            self.in_constrained=[False,False]
            self.turn_speed=[0,5]
            
            return self.route,self.turns,self.edges_list,self.next_turn_point,self.groups,self.in_constrained,self.turn_speed
        
        route=None
        turns=None
        groups=None
        edges_list=None
        next_turn_point=None

        self.start_point=Point(tuple((lon,lat)))
        self.start_index=next_node_index
        self.start_index_previous=prev_node_osmnx_id
        if prev_node_osmnx_id>4480 and prev_node_osmnx_id!=5000:
            prev_node_osmnx_id=5000
            self.start_index=self.start_index_previous
            next_node_index=self.start_index_previous
            self.start_index_previous=prev_node_osmnx_id

        ## check for changes in the aircrafts subgraph if any
        expanded=False
        change_list=[]
        
        replan_bool=True
        
        k_list=[]

        for change in changes_list:
            
            
            k=change[0]
            k_list.append(k)
            kk=change[1]
            if k==prev_node_osmnx_id and kk==next_node_index:
                #if the current edge is set to 0 speed  or if high traffic geofenc eis set and you are not of high priority do nothing
                if change[2]==0 or (change[2]>0 and change[2]<1 and self.priority<3):
                    ##If the current group is set to a zero speed geofence
                    replan_bool=False
                continue
            
            if self.loitering:
                #if you are a loitering mission do not take into account your own geofence
                if (k,kk) in self.loitering_edges:
                    continue
            
            
            expanded=False
            
##########################

            if k not in self.graph.key_indices_list or kk not in self.graph.key_indices_list:
                continue              
            result = np.where(self.os_keys2_indices ==k)
            rr=np.where(result[1] ==0)

            for ii in self.os_keys2_indices[result[0][rr]][0][1:]:
                if ii==65535:
                    break
                for p in self.graph.children_list[ii]:
                    if p==65535:
                        break
                    if kk==self.graph.key_indices_list[p]:
                           change_list.append([ii,p])
                           break
###########    

        if change_list==[]:
            replan_bool=False
              
        if prev_node_osmnx_id<4481 and replan_bool and next_node_index<4481:

                # Do not replan in high traffic if you have low priority, should the same happen when in loitering mission?
                #TODO check if teh second condition is not needed
            if (self.flow_graph.edges_current_speed[prev_node_osmnx_id][next_node_index]<1 and self.flow_graph.edges_current_speed[prev_node_osmnx_id][next_node_index]!=0 and self.priority<3):# or self.edge_gdf[prev_node_osmnx_id][next_node_index].speed==0:
                replan_bool=False
            
        if not replan_bool and change_list!=[]:
            self.update_changed_vertices(self.path,self.graph,self.flow_graph.edges_current_speed,self.flow_graph.edges_graph,self.flow_graph.edges_previous_speed,True,change_list)


        if change_list!=[] and replan_bool:
            
# =============================================================================
#             if (str(next_node_index)+'-'+str(prev_node_osmnx_id)) in self.os_keys_dict_pred.keys():
#                 start_id=self.os_keys_dict_pred[str(next_node_index)+'-'+str(prev_node_osmnx_id)]
#             else:
#                 prev_node_osmnx_id =0
#                 start_id=self.os_keys_dict_pred[str(next_node_index)+'-'+str(prev_node_osmnx_id)]
# =============================================================================
 ############## 
            start_id=None
            result = np.where(self.os_keys2_indices ==self.start_index)
            rr=np.where(result[1] ==0)
            if self.start_index_previous==5000:
                start_id=self.os_keys2_indices[result[0][rr]][0][1]
            else:
                for ii in self.os_keys2_indices[result[0][rr]][0][1:]:
                    if ii==65535:
                        break
                    for p in self.graph.parents_list[ii]:
                        if p ==65535:
                            break
                        if self.start_index_previous==self.graph.key_indices_list[p]:
                            start_id=ii
                            break
                    if start_id !=None:
                        break
  #####################################            
# =============================================================================
#             start_id=None
#             for i in self.os_keys2_indices:
#                 key=i[0]
#                 if key==self.start_index:
#                     if self.start_index_previous==0:
#                         start_id=i[1]
#                         break
#                     
#                     for ii in i[1:]:
#                         if ii==65535:
#                             break
#                         for p in self.graph.parents_list[ii]:
#                             if p ==65535:
#                                 break
#                             if self.start_index_previous==self.graph.key_indices_list[p]:
#                                 start_id=ii
#                                 break
#                         if start_id !=None:
#                             break                      
#                 if start_id !=None :
#                     break
# =============================================================================
 

        
 
                
            start_node=start_id
            self.path.start=start_node
            

            self.start_index=next_node_index
            self.start_index_previous=prev_node_osmnx_id

            ##call get path
            route,turns,indices_nodes,turn_coord,groups,in_constrained,turn_speed,init_groups=self.get_path(self.path,self.graph,self.flow_graph.edges_current_speed,self.flow_graph.edges_graph,self.flow_graph.edges_previous_speed,True,change_list)
            
            self.path.origin_node_index=start_id
             
            if route != None :
                edges_list=[]
                next_turn_point=[]
                os_id1=self.start_index_previous

                os_id2=indices_nodes[0]

                
                if 2000 not in init_groups and self.start_index_previous==5000:
                    edges_list.append((os_id1,os_id2))

                    nodes_index=0

                    
                    for i in range(len(init_groups)-1):
                        edges_list.append((os_id1,os_id2))

                        if nodes_index>len(init_groups)-2:
                            break
                        if nodes_index<len(init_groups)-1 and indices_nodes[nodes_index+1]==os_id2:
                            nodes_index=nodes_index+1
                            continue
    
                        nodes_index=nodes_index+1
                        os_id1=os_id2
                        os_id2=indices_nodes[nodes_index]
                
                else:
                    if not( init_groups[0]==2000 and init_groups[2]!=2000) :
                        edges_list.append((os_id1,os_id2))
                    
                    nodes_index=1
    
                    for i in range(len(init_groups)-1):
                        edges_list.append((os_id1,os_id2))
                        if nodes_index>len(init_groups)-2:
                            break
                        if nodes_index<len(init_groups)-1 and indices_nodes[nodes_index+1]==os_id2:
                            nodes_index=nodes_index+1
                            continue
                        if nodes_index<len(init_groups)-2 and init_groups[nodes_index+1]==2000 and init_groups[nodes_index+2]!=2000:
                            nodes_index=nodes_index+1
                            os_id2=indices_nodes[nodes_index]
    
                        if init_groups[nodes_index]==2000 and init_groups[nodes_index+1]==2000:
                            nodes_index=nodes_index+1
                            os_id1=5000
                            os_id2=indices_nodes[nodes_index]
    
                        else:
                            nodes_index=nodes_index+1
                            os_id1=os_id2
                            os_id2=indices_nodes[nodes_index]


                            
                cnt=0
                for i in turns:
                    if i:
                        next_turn_point.append(turn_coord[cnt])
                        cnt=cnt+1
                    else:
                        next_turn_point.append(turn_coord[cnt])
                                
                del indices_nodes[0]
                
                turns[-1]=True
                turn_speed[-1]=5
                            
                self.route=np.array(route,dtype=np.float64)
                self.turns=np.array(turns,dtype=np.bool8) 
                self.edges_list=np.array(edges_list) 
                self.next_turn_point=next_turn_point
                self.groups=np.array(groups,dtype=np.uint16)
                self.in_constrained=np.array(in_constrained,dtype=np.bool8)
                self.turn_speed=np.array(turn_speed,dtype=np.float64)
                return self.route,self.turns,self.edges_list,self.next_turn_point,self.groups,self.in_constrained,self.turn_speed
            

        return [],[],[],[],[],[],[]
      
    ##Function handling the replanning process, called when aircraft is spawned
    ##Returns: route,turns,edges_list,next_turn_point,groups,in_constrained,turn_speed
    ##route is the list of waypoints (lon,lat)
    ##turns is the list of booleans indicating for every waypoint if it is a turn
    ##edges_list is the list of the edges
    ##next_turn_point is a list containing the coord of every point that is a turn point
    ##groups is the list of the group in which each waypoint belongs to
    ##in_constrained is the list of booleans indicating for every waypoint if it is in constarined airspace
    ##turn_speed is teh list if speed to be used if the waypoint is a turning waypoint  
    def replan_spawned(self,changes_list,prev_node_osmnx_id,next_node_index,lat,lon):
        if self.in_same_cell:
            self.route=[(self.start_point.x,self.start_point.y),(self.goal_point.x,self.goal_point.y)]
            self.turns=[False,True]
            self.edges_list=[(self.start_index_previous,self.start_index),(self.goal_index_next,self.goal_index)]
            self.next_turn_point=[(-999,-999),(-999,-999)]
            self.groups=[2000,2000]    
            self.in_constrained=[False,False]
            self.turn_speed=[0,1]
            
            return self.route,self.turns,self.edges_list,self.next_turn_point,self.groups,self.in_constrained,self.turn_speed

        route=None
        turns=None
        groups=None
        edges_list=None
        next_turn_point=None

        
        self.start_point=Point(tuple((lon,lat)))
        

        ## check for changes in the aircrafts subgraph if any

        expanded=False
        change_list=[]
        
        replan_bool=True
        

        k_list=[]
        for change in changes_list:

            k=change[0]
            k_list.append(k)
            kk=change[1]
            if k==prev_node_osmnx_id and kk==next_node_index:
                #if the current edge is set to 0 speed  or if high traffic geofenc eis set and you are not of high priority do nothing
                if change[2]==0 or (change[2]>0 and change[2]<1 and self.priority<3):
                    ##If the current group is set to a zero speed geofence
                    replan_bool=False
                continue
            
            if self.loitering:
                #if you are a loitering mission do not take into account your own geofence
                if (k,kk) in self.loitering_edges:
                    continue
######################
            if k not in self.graph.key_indices_list or kk not in self.graph.key_indices_list:
                continue    
            result = np.where(self.os_keys2_indices ==k)
            rr=np.where(result[1] ==0)
            for ii in self.os_keys2_indices[result[0][rr]][0][1:]:
                if ii==65535:
                    break
                for p in self.graph.children_list[ii]:
                    if p ==65535:
                           break
                    if kk==self.graph.key_indices_list[p]:
                           change_list.append([ii,p])
                           break
#######################
        if change_list==[]:
            replan_bool=False
# =============================================================================
#         expanded=False
#         for i in self.os_keys2_indices:
#             key=i[0]
#             while key in k_list:
#                 ind=k_list.index(key)
#                 kk=changes_list[k_list.index(key)]
#                 b=False
#                 for ii in i[1:]:
#                     if ii==65535:
#                         break
#                     for ch in self.graph.children_list[ii]:
#                         if ch==65535:
#                             break
#                         if kk==self.graph.key_indices_list[ch]:
#                             if self.graph.expanded_list[ch] or self.graph.expanded_list[ii]:
#                                 change_list.append([ii,ch])
#                                 b=True
# 
#                             break
#                     if b:
#                         break
#                     
#                 del k_list[ind]
#                 del changes_list[ind]
# 
#             if len(k_list)==0:
#                 break 
# 
# =============================================================================
                
        if prev_node_osmnx_id<4481 and replan_bool and next_node_index<4481 :

                   # Do not replan in high traffic if you have low priority, should the same happen when in loitering mission?
            if (self.flow_graph.edges_current_speed[prev_node_osmnx_id][next_node_index]<1 and self.flow_graph.edges_current_speed[prev_node_osmnx_id][next_node_index]!=0 and self.priority<3):# or self.edge_gdf[prev_node_osmnx_id][next_node_index].speed==0:
                replan_bool=False

            
        if not replan_bool and change_list!=[]:
            self.update_changed_vertices(self.path,self.graph,self.flow_graph.edges_current_speed,self.flow_graph.edges_graph,self.flow_graph.edges_init_speed,True,change_list)
            
        if change_list!=[] and replan_bool:
            
# =============================================================================
#             if (str(next_node_index)+'-'+str(prev_node_osmnx_id)) in self.os_keys_dict_pred.keys():
#                 start_id=self.os_keys_dict_pred[str(next_node_index)+'-'+str(prev_node_osmnx_id)]
#             else:
#                 prev_node_osmnx_id =0
#                 start_id=self.os_keys_dict_pred[str(next_node_index)+'-'+str(prev_node_osmnx_id)]
#             start_node=self.flow_graph.nodes_graph[self.graph[start_id].key_index]
#             
# =============================================================================

#########################
            start_id=None
            result = np.where(self.os_keys2_indices ==self.start_index)
            rr=np.where(result[1] ==0)
            if self.start_index_previous==5000:
                start_id=self.os_keys2_indices[result[0][rr]][0][1]
            else:
                for ii in self.os_keys2_indices[result[0][rr]][0][1:]:
                    if ii==65535:
                        break
                    for p in self.graph.parents_list[ii]:
                        if p ==65535:
                            break
                        if self.start_index_previous==self.graph.key_indices_list[p]:
                            start_id=ii
                            break
                    if start_id !=None:
                        break
#####################
# =============================================================================
#            start_id=None
#             for i in self.os_keys2_indices:
#                 key=i[0]
#                 if key==self.start_index:
#                     if self.start_index_previous==0:
#                         start_id=i[1]
#                         break                    
#                     for ii in i[1:]:
#                         if ii==65535:
#                             break
#                         for p in self.graph.parents_list[ii]:
#                             if p ==65535:
#                                 break
#                             if self.start_index_previous==self.graph.key_indices_list[p]:
#                                 start_id=ii
#                                 break
#                         if start_id !=None:
#                             break                      
#                 if start_id !=None :
#                     break
# =============================================================================
                
            start_node=start_id
            self.path.start=start_node
            
# =============================================================================
#             self.start_index=next_node_index
#             self.start_index_previous=prev_node_osmnx_id
# =============================================================================


            ##call get path
            route,turns,indices_nodes,turn_coord,groups,in_constrained,turn_speed,init_groups=self.get_path(self.path,self.graph,self.flow_graph.edges_current_speed,self.flow_graph.edges_graph,self.flow_graph.edges_init_speed,True,change_list)
            self.path.origin_node_index=start_id
             
            if route != None :
                edges_list=[]
                next_turn_point=[]
                os_id1=self.start_index_previous

                os_id2=indices_nodes[0]
            
                
                if 2000 not in init_groups and self.start_index_previous==5000:
                    edges_list.append((os_id1,os_id2))

                    nodes_index=0

                    
                    for i in range(len(init_groups)-1):
                        edges_list.append((os_id1,os_id2))

                        if nodes_index>len(init_groups)-2:
                            break
                        if nodes_index<len(init_groups)-1 and indices_nodes[nodes_index+1]==os_id2:
                            nodes_index=nodes_index+1
                            continue
    
                        nodes_index=nodes_index+1
                        os_id1=os_id2
                        os_id2=indices_nodes[nodes_index]
                
                else:
                    if not( init_groups[0]==2000 and init_groups[2]!=2000) :
                        edges_list.append((os_id1,os_id2))
                    
                    nodes_index=1
    
                    for i in range(len(init_groups)-1):
                        edges_list.append((os_id1,os_id2))
                        if nodes_index>len(init_groups)-2:
                            break
                        if nodes_index<len(init_groups)-1 and indices_nodes[nodes_index+1]==os_id2:
                            nodes_index=nodes_index+1
                            continue
                        if nodes_index<len(init_groups)-2 and init_groups[nodes_index+1]==2000 and init_groups[nodes_index+2]!=2000:
                            nodes_index=nodes_index+1
                            os_id2=indices_nodes[nodes_index]
    
                        if init_groups[nodes_index]==2000 and init_groups[nodes_index+1]==2000:
                            nodes_index=nodes_index+1
                            os_id1=5000
                            os_id2=indices_nodes[nodes_index]
    
                        else:
                            nodes_index=nodes_index+1
                            os_id1=os_id2
                            os_id2=indices_nodes[nodes_index]

                            
                cnt=0
                for i in turns:
                    if i:
                        next_turn_point.append(turn_coord[cnt])
                        cnt=cnt+1
                    else:
                        next_turn_point.append(turn_coord[cnt])
                                
                del indices_nodes[0]
                        

                turns[-1]=True
                turn_speed[-1]=5
                            
                self.route=np.array(route,dtype=np.float64)
                self.turns=np.array(turns,dtype=np.bool8) 
                self.edges_list=np.array(edges_list) 
                self.next_turn_point=next_turn_point
                self.groups=np.array(groups,dtype=np.uint16)
                self.in_constrained=np.array(in_constrained,dtype=np.bool8)
                self.turn_speed=np.array(turn_speed,dtype=np.float64)
                return self.route,self.turns,self.edges_list,self.next_turn_point,self.groups,self.in_constrained,self.turn_speed
            
            

        return self.route,self.turns,self.edges_list,self.next_turn_point,self.groups,self.in_constrained,self.turn_speed