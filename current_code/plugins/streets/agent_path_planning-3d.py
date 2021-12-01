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


class CellNode:
    def __init__(self,cell):
        self.p0=cell.p0
        self.p1=cell.p1
        self.p2=cell.p2
        self.p3=cell.p3

class Node:
    av_speed_horizontal= 10.0#10.0 ##TODO: that needs fine tunng
    av_speed_vertical=1.0
    def __init__(self,key_index,lon,lat,index,group):
        self.key_index=key_index # the index the osmnx graph
        self.index=index# the index in the search graph

        #the coordinates of the node as given by osmnx (latitude,longitude)
        ##Coordinates of the center
        self.lon=lon
        self.lat=lat
        self.z=0
        
        ##Coordinates of the center
        self.x_cartesian=None
        self.y_cartesian=None


        #the parents(predessecors) and children(successor) of the node expressed as lists containing their indexes in the graph 
        self.parents=[]
        self.children=[]
        
        #self.f=0.0
        self.g=float('inf')
        self.rhs=float('inf')
        self.key=[0.0,0.0]
 
        self.inQueue=False
        
        #the stroke group
        self.group=group
        
        self.expanded=False
        
        self.open_airspace=False
        self.cell=None
        
class Path:
    def __init__(self,start,goal,speed):
        self.start=start
        self.goal=goal
        self.k_m=0
        self.queue=[]
        self.origin_node_index=None
        self.speed=speed
        
##Initialise the path planning      
def initialise(path):
    path.queue=[]
    path.k_m=0
    path.goal.rhs=0
    path.goal.inQueue=True
    path.goal.h=heuristic(path.start,path.goal,path.speed)
    path.goal.expanded=True
    heapq.heappush(path.queue, (path.goal.h,0,path.goal.index, path.goal))
    path.origin_node_index=path.start.index
 

##Compare the keys of two nodes
def compare_keys(node1,node2):
    if node1[0]<node2[0]:
        return True
    elif node1[0]==node2[0] and node1[1]<node2[1]:
        return True
    return False
    
##Calculate the keys of a node    
def calculateKey(node,start, path):
    return (min(node.g, node.rhs) + heuristic(node,start,path.speed) + path.k_m, min(node.g, node.rhs))

##Calculate the distance of two points in cartesian coordinates
def eucledean_distance(p1,p2):
    return  math.sqrt((p1.x_cartesian-p2.x_cartesian)*(p1.x_cartesian-p2.x_cartesian)+ (p1.y_cartesian-p2.y_cartesian)*(p1.y_cartesian-p2.y_cartesian) )    

def heuristic(current, goal,speed):
    if current.open_airspace or goal.open_airspace:
        h=eucledean_distance(current, goal)/speed
        h=h+abs(current.z-goal.z)/current.av_speed_vertical
    else:
        h=(abs(current.x_cartesian-goal.x_cartesian)+abs(current.y_cartesian-goal.y_cartesian))/min(current.av_speed_horizontal,speed)

    if current.group!=goal.group:
        h=h+9.144/current.av_speed_vertical
    return h


##Compute the cost of moving from a node to its neighborh node
def compute_c(current,neigh,edges,speed):
    g=1
    if current.open_airspace  or neigh.open_airspace:
        g=eucledean_distance(current,neigh)/speed
        g=g+abs(current.z-neigh.z)/current.av_speed_vertical
    else:
        if(current.group!=neigh.group):
            g=9.144/current.av_speed_vertical
        else:
            #check if the group is changing (the drone needs to turn)
            if current.group==neigh.group:
                if edges[current.key_index][neigh.key_index].speed==0:
                    g=float('inf')
    
                else:
                    g=edges[current.key_index][neigh.key_index].length/min(edges[current.key_index][neigh.key_index].speed,speed)
    return g    

##Return the top key of the queue 
def top_key(q):
    if len(q) > 0:
        return [q[0][0],q[0][1]]
    else:
        print('empty queue!')
        return [float('inf'), float('inf')]
    
##Update the vertex of a node
def update_vertex(path,node):

    if node.g!=node.rhs and node.inQueue:     
        #Update
        id_in_queue = [item for item in path.queue if node.index==item[2]]
        if id_in_queue != []:
            if len(id_in_queue) != 1:
                raise ValueError('more than one ' + str(node.key_index) + ' in the queue!')
            node.key=calculateKey(node, path.start, path)
            path.queue[path.queue.index(id_in_queue[0])]=path.queue[-1]
            path.queue.pop()
            heapq.heapify(path.queue)
            heapq.heappush(path.queue, (node.key[0],node.key[1],node.index,node))
            
    elif node.g!=node.rhs and (not node.inQueue):
        #Insert
        node.inQueue=True
        node.key=calculateKey(node, path.start, path)
        heapq.heappush(path.queue, (node.key[0],node.key[1],node.index,node))
        
    elif node.g==node.rhs and node.inQueue: 
        #remove
        id_in_queue = [item for item in path.queue if node.index==item[2]]
        
        if id_in_queue != []:
            if len(id_in_queue) != 1:
                raise ValueError('more than one ' + id + ' in the queue!')
            node.inQueue=False
            path.queue[path.queue.index(id_in_queue[0])]=path.queue[-1]
            path.queue.pop()
            heapq.heapify(path.queue)
          

          
##Compute the shortest path using D* Lite
##returns flase if no path was found
def compute_shortest_path(path,graph,edges):

    path.start.key=calculateKey(path.start, path.start, path)
    k_old=top_key(path.queue)
   
    while path.start.rhs>path.start.g or compare_keys(k_old,path.start.key):

        if len(path.queue)==0:
            print("No path found!")
            return 0

        k_old=top_key(path.queue)
        current_node=path.queue[0][3]#get the node with the highest priority
        current_node.expanded=True

        
        k_new=calculateKey(current_node, path.start, path)
        
        if compare_keys(k_old, k_new):
            heapq.heappop(path.queue)
            current_node.key=k_new
            current_node.inQueue=True
            current_node.expanded=True
            heapq.heappush(path.queue, (current_node.key[0],current_node.key[1],current_node.index,current_node))
            
        elif current_node.g>current_node.rhs:
            current_node.g=current_node.rhs
            heapq.heappop(path.queue)
            current_node.inQueue=False

            for p in current_node.parents:
                pred_node=graph[p]  

                if pred_node!=path.goal:
                    pred_node.rhs=min(pred_node.rhs,current_node.g+compute_c(pred_node,current_node,edges,path.speed))
                update_vertex(path, pred_node)
        else:
            g_old=copy.deepcopy(current_node.g)
            current_node.g=float('inf')
            pred_node=current_node

                    
            for p in current_node.parents:
                parent=graph[p]
                if parent.rhs==(g_old+compute_c(parent,current_node,edges,path.speed)):
                    if(parent!=path.goal):
                        tt=[]
                        for ch in parent.children:
                            child=graph[ch]
                            tt.append(child.g+compute_c(parent,child,edges,path.speed))
                        parent.rhs=min(tt)
                update_vertex(path, parent)
            if pred_node.rhs==g_old:
                if pred_node!= path.goal:
                    tt=[]
                    for ch in pred_node.children:
                        child=graph[ch]
                        tt.append(child.g+compute_c(pred_node,child,edges,path.speed))
                    pred_node.rhs=min(tt)
            update_vertex(path, pred_node)               

        k_old=top_key(path.queue)
        path.start.key=calculateKey(path.start, path.start, path)
        

    path.start.g=path.start.rhs
            
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



class PathPlanning:
    
    def __init__(self,aircraft_type,priority,open_airspace_grid,flow_control_graph,gdf,lon_start,lat_start,lon_dest,lat_dest,loitering=False,loitering_edges=[]):
        self.aircraft_type=aircraft_type
        self.start_index=None
        self.start_index_previous=None
        self.start_in_open=True
        self.goal_index=None
        self.goal_index_next=None
        self.dest_in_open=True
        self.open_airspace_grid=open_airspace_grid
        self.flow_control_graph=copy.deepcopy(flow_control_graph)
        self.gdf=gdf
        self.G = None
        self.edge_gdf=None
        self.path=None
        self.os_keys_dict_succ={}
        self.os_keys_dict_pred={}
        self.route=[]
        self.turns=[]
        self.priority=priority #4,3,2,1 in decreasing priority
        self.loitering=loitering
        if self.loitering:
            self.loitering_edges=loitering_edges

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
            if distance<0.00001:
                self.start_index=v
                self.start_index_previous=u
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
                    self.start_index_previous=0


        
        if self.dest_in_open:
            point=(lat_dest,lon_dest)
            geometry, u, v,distance=get_nearest_edge(self.gdf, point)
            if distance<0.00001:
                self.goal_index=u
                self.goal_index_next=v
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
                    self.goal_index_next=0

        del self.open_airspace_cells
        
        
            
        if self.goal_index_next==self.start_index and self.goal_index==self.start_index_previous:
            print("same goal to start index")
            
            
        #find the area of interest based on teh start and goal point
        ##TODO: tune the exp_const
        if not self.start_in_open and not self.dest_in_open:
            exp_const=0.02##0.005 
            box=bbox(min(lat_start,lat_dest)-exp_const,min(lon_start,lon_dest)-exp_const,max(lat_start,lat_dest)+exp_const,max(lon_start,lon_dest)+exp_const) 
            
            G,edges=self.flow_control_graph.extract_subgraph(box)
            self.G=copy.deepcopy(G)
            self.edge_gdf=copy.deepcopy(edges)
        else:
            exp_const=0.02##0.005 
            box=bbox(min(self.start_point.y,self.goal_point.y)-exp_const,min(self.start_point.x,self.goal_point.x)-exp_const,max(self.start_point.y,self.goal_point.y)+exp_const,max(self.start_point.x,self.goal_point.x)+exp_const) 
    
            G,edges=self.flow_control_graph.extract_subgraph(box)
            self.G=copy.deepcopy(G)
            self.edge_gdf=copy.deepcopy(edges)
            
        
        del self.flow_control_graph #empty these, we do not need it any more
        del self.gdf

        #Create the graph
        self.graph=[]

        connected2open=False
        omsnx_keys_list=list(self.G.keys())
        
        transformer = Transformer.from_crs('epsg:32633', 'epsg:4326')
        #Add open airspace nodes to graph
        for i in range(len(self.open_airspace_grid.grid)):
           cell=self.open_airspace_grid.grid[i]
           group=-1
           x=cell.center_x
           y=cell.center_y
           z=cell.minimum_altitude
           p=transformer.transform(x,y)
           lon=p[1]
           lat=p[0]
           key=cell.key_index
           dict_tmp={}
           dict_tmp[0]=i
           self.os_keys_dict_pred[key]=dict_tmp
           self.os_keys_dict_succ[key]=dict_tmp
           node=Node(key,lon,lat,i,group)
           node.open_airspace=True
           node.cell=CellNode(cell)
           node.x_cartesian=cell.center_x
           node.y_cartesian=cell.center_y
           node.z=z
           for j in cell.neighbors:
               node.children.append(j)
               
           if not connected2open:
               for k in cell.entry_list:
                   if k in omsnx_keys_list:
                       connected2open=True
                       break
           if not connected2open:
               for k in cell.exit_list:
                   if k in omsnx_keys_list:
                       connected2open=True
                       break

           self.graph.append(node)

            
        
           
        for node in self.graph:      
            neighboors=copy.deepcopy(node.children)
            node.children=[]
            for i in neighboors:
                key=self.os_keys_dict_succ[i][0]
                node.children.append(key)
                node.parents.append(key)
                
        ##If there is no open airspace cell coneccted to the extracted constarined do not use open
        if not connected2open:
            self.graph=[]
            self.os_keys_dict_succ={}
            self.os_keys_dict_pred={}

        ##Add constrained nodes to graph
        #omsnx_keys_list=list(self.G.keys())
        transformer = Transformer.from_crs( 'epsg:4326','epsg:32633')
             
        
        new_nodes_counter=0
        graph_len=len(self.graph)
        for i in range(len(omsnx_keys_list)):
           key=omsnx_keys_list[i]
           lon=self.G[key].x
           lat=self.G[key].y 

           parents=self.G[key].parents
           children=self.G[key].children
           my_group={} 
        
           ii=0
           tmp=[]#list if the groups that the node has been added
           for p in parents:
               if not ii:
                   if p in list(self.edge_gdf.keys()) and key in self.edge_gdf[p]: 

                       group=int(self.edge_gdf[p][key].stroke_group)
                       node=Node(key,lon,lat,i+new_nodes_counter+graph_len,group)
                       pp=transformer.transform(lat,lon)
                       node.x_cartesian,node.y_cartesian =pp[0],pp[1]
                       my_group.update({i+new_nodes_counter+graph_len:group})
                       self.graph.append(node)
                       tmp.append(group)
                       ii=ii+1
                       if key in self.os_keys_dict_pred.keys():
                           self.os_keys_dict_pred[key][p]=i+new_nodes_counter+graph_len
                       else:
                           dict={}
                           dict[p]=i+new_nodes_counter+graph_len
                           self.os_keys_dict_pred[key]=dict
               else: 
                if p in list(self.edge_gdf.keys()) and key in self.edge_gdf[p]: 

                        new_nodes_counter=new_nodes_counter+1
                        group=int(self.edge_gdf[p][key].stroke_group)
                        node=Node(key,lon,lat,i+new_nodes_counter+graph_len,group)
                        pp=transformer.transform(lat,lon)
                        node.x_cartesian,node.y_cartesian = pp[0],pp[1]
                        my_group.update({i+new_nodes_counter+graph_len:group})
                        self.graph.append(node)
                        tmp.append(group)
                        ii=ii+1
                        if key in self.os_keys_dict_pred.keys():
                           self.os_keys_dict_pred[key][p]=i+new_nodes_counter+graph_len
                        else:
                           dict={}
                           dict[p]=i+new_nodes_counter+graph_len
                           self.os_keys_dict_pred[key]=dict
                           
           for ch in children:
                group=int(self.edge_gdf[key][ch].stroke_group)
                if not group in tmp:
                    if not ii:
                        node=Node(key,lon,lat,i+new_nodes_counter+graph_len,group)
                        p=transformer.transform(lat,lon)
                        node.x_cartesian,node.y_cartesian = p[0],p[1]
                        my_group.update({i+new_nodes_counter+graph_len:group})
                        
                        self.graph.append(node)
                        tmp.append(group)
                        ii=ii+1
                        if key in self.os_keys_dict_succ.keys():
                           self.os_keys_dict_succ[key][ch]=i+new_nodes_counter+graph_len
                        else:
                           dict={}
                           dict[ch]=i+new_nodes_counter+graph_len
                           self.os_keys_dict_succ[key]=dict
                        
                    else:
                        new_nodes_counter=new_nodes_counter+1
                        node=Node(key,lon,lat,i+new_nodes_counter+graph_len,group)
                        p=transformer.transform(lat,lon)
                        node.x_cartesian,node.y_cartesian = p[0],p[1]
                        my_group.update({i+new_nodes_counter+graph_len:group})
                        self.graph.append(node)
                        tmp.append(group)
                        ii=ii+1
                        if key in self.os_keys_dict_succ.keys():
                           self.os_keys_dict_succ[key][ch]=i+new_nodes_counter+graph_len
                        else:
                           dict={}
                           dict[ch]=i+new_nodes_counter+graph_len
                           self.os_keys_dict_succ[key]=dict
                        
           if ii==0:
               #continue
                node=Node(key,lon,lat,i+new_nodes_counter+graph_len,-1)
                p=transformer.transform(lat,lon)
                node.x_cartesian,node.y_cartesian =p[0],p[1]
                self.graph.append(node)

                self.os_keys_dict_succ[key]=i+new_nodes_counter+graph_len
                self.os_keys_dict_pred[key]=i+new_nodes_counter+graph_len

                #print("No succ or pred: "+str(key))

                        
           if len(my_group)>1:
               for index in my_group:
                   for index_ in my_group:
                        if my_group[index]!=my_group[index_] and index!=index_:
                            self.graph[index].children.append(index_)
                            self.graph[index].parents.append(index_)     
                            
        #add the children and parents to each node                
        for ii,i in enumerate(self.graph):
            if ii<graph_len:
                cell=self.open_airspace_grid.grid[ii]
                for ch in cell.entry_list:
                    for j in self.graph:
                        if ch==j.key_index:
                            i.children.append(j.index)
                            j.parents.append(i.index)

                            
                            if i.key_index in self.os_keys_dict_succ.keys():
                                    self.os_keys_dict_succ[i.key_index][ch]=i.index
                            else:
                                    dict={}
                                    dict[p]=i.index
                                    self.os_keys_dict_succ[i.key_index]=dict   
                                    
                            if j.key_index in self.os_keys_dict_pred.keys():
                                    self.os_keys_dict_pred[j.key_index][i.key_index]=j.index
                            else:
                                    dict={}
                                    dict[i.key_index]=j.index
                                    self.os_keys_dict_pred[j.key_index]=dict 
                            break                    
                for p in cell.exit_list:
                    for j in self.graph:
                        if p==j.key_index :
                            i.parents.append(j.index)
                            j.children.append(i.index)

                            if i.key_index in self.os_keys_dict_pred.keys():
                                    self.os_keys_dict_pred[i.key_index][p]=i.index
                            else:
                                    dict={}
                                    dict[p]=i.index
                                    self.os_keys_dict_pred[i.key_index]=dict   
                                    
                            if j.key_index in self.os_keys_dict_succ.keys():
                                    self.os_keys_dict_succ[j.key_index][i.key_index]=j.index
                            else:
                                    dict={}
                                    dict[i.key_index]=j.index
                                    self.os_keys_dict_succ[j.key_index]=dict 
                            break
            else:
                key=i.key_index
                parents=self.G[key].parents
                children=self.G[key].children
                for p in parents:
                    for j in self.graph:
                        if p==j.key_index and (j.group==i.group or i.group==-1) :
                            i.parents.append(j.index)
                            
                            if i.key_index in self.os_keys_dict_pred.keys():
                                self.os_keys_dict_pred[i.key_index][p]=i.index
                            else:
                                dict={}
                                dict[p]=i.index
                                self.os_keys_dict_pred[i.key_index]=dict                        
                            
                            break
              
                for ch in children:
                    for j in self.graph:
                        if ch==j.key_index and (j.group==i.group or i.group==-1):
                            i.children.append(j.index)
                            
                            if i.key_index in self.os_keys_dict_succ.keys():
                                    self.os_keys_dict_succ[i.key_index][ch]=i.index
                            else:
                                    dict={}
                                    dict[ch]=i.index
                                    self.os_keys_dict_succ[i.key_index]=dict                        
                            break
                        
        del self.open_airspace_grid
        del self.G
        

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

        start_id=self.os_keys_dict_pred[self.start_index][self.start_index_previous]
        goal_id=self.os_keys_dict_succ[self.goal_index][self.goal_index_next]
        
        start_node=self.graph[start_id] 
        x_start=start_node.lon
        y_start=start_node.lat
        
        goal_node=self.graph[goal_id] 
        
        x_goal=goal_node.lon
        y_goal=goal_node.lat
        
        self.path=Path(start_node,goal_node,self.speed_max)
        
        initialise(self.path)
        
        path_found=compute_shortest_path(self.path,self.graph,self.edge_gdf)
        print(path_found)
        
        route=[]
        turns=[]
        edges_list=[]
        next_turn_point=[]
        indices_nodes=[]
        turn_indices=[]
        if path_found:
            route,turns,indices_nodes,turn_coord,groups,in_constrained,turn_speed=self.get_path(self.path,self.graph,self.edge_gdf)

            os_id1=self.start_index_previous

            os_id2=indices_nodes[0]
            cnt=0
            for i in range(0,len(indices_nodes)-1):
                
                    
                if indices_nodes[i]==-1 or indices_nodes[i]==os_id2:
                    cnt=cnt+1
                else:
                    for j in range(cnt):
                        edges_list.append((os_id1,os_id2))
                    #edges_list.append((os_id1,indices_nodes[i]))
                    cnt=1
                    os_id1=os_id2
                    os_id2=indices_nodes[i]
                    if i>0 and groups[i]==-1 and groups[i-1]==-1:
                        os_id1=0
                        
                if i>0 and groups[i]!=-1 and groups[i-1]==-1:
                    #print(i)
                    os_id2=indices_nodes[i]
                    del edges_list[-1]
                    #continue
            for j in range(cnt):
                edges_list.append((os_id1,os_id2))
                
            if self.goal_index_next!=0:
                edges_list.append((self.goal_index,self.goal_index_next))
            else:
                edges_list.append((0,self.goal_index))
                
            cnt=0
            for i in range(len(turns)):
                if turns[i]:# and in_constrained[i]:
                    next_turn_point.append(turn_coord[cnt])
                    cnt=cnt+1
                else:
                    next_turn_point.append(turn_coord[cnt])
                    
        del indices_nodes[0]
                    
        self.route=route
        self.turns=turns 
        self.edges_list=edges_list
        self.next_turn_point=next_turn_point
        self.groups=groups    
        self.in_constrained=in_constrained
        self.turn_speed=turn_speed
        
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
    def get_path(self,path,graph, edges,edges_old=None,change=False,change_list=[]):

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
            path.k_m=path.k_m+heuristic(graph[path.origin_node_index],path.start,path.speed)
            for c in change_list:
                
                if not  c[0].expanded or not c[1].expanded:
                    print("not expanded")
                    

                c_old=compute_c(c[0], c[1],edges_old,path.speed)

                    #update cost and obstacles here
                if c_old>compute_c(c[0], c[1],edges,path.speed): #if cost is decreased

                    if(c[0]!=path.goal):
    
                        c[0].rhs=min(c[0].rhs,compute_c(c[0], c[1], edges,path.speed)+c[1].g)

                            
                elif c[0].rhs== c_old+c[1].g: #if cost is increased

                    if c[0]!=path.goal:
                        tt=[]
                        for ch in c[0].children:
                            child=graph[ch]
                            tt.append(child.g+compute_c( c[0],child, edges,path.speed))
                        c[0].rhs=min(tt)
                        path.start.rhs=float('inf')## not sure for that

                update_vertex(path, c[0])
                
                edges_old[c[0].key_index][c[1].key_index].speed=edges[c[0].key_index][c[1].key_index].speed
            path_found=compute_shortest_path(path,graph,edges_old)


            print(path_found)
            change=False 
            if not  path_found:
                print("Compute path all from the start")
                #break
                
        if not  path_found:
            for i in range(len(graph)):
                n=graph[i]
                n.g=float('inf')
                n.rhs=float('inf')
                n.inQueue=False
                n.expanded=False
                n.h=0.0
                n.key=[0.0,0.0]
                self.route_origin_node=copy.deepcopy(path.start)
                    
            initialise(path)
            path_found=compute_shortest_path(path,graph,edges) 
            
        if not path_found:
            return None,None,None,None,None,None,None
        

        next_node_index.append(self.start_index)
        tmp=(self.start_point.x,self.start_point.y)
        group_numbers.append(path.start.group)
        route_centers.append(tmp)
        turns.append(0)
        
        if path.start.open_airspace:
            next_node_index.append(self.start_index)
            tmp=(self.start_point.x,self.start_point.y)
            group_numbers.append(path.start.group)
            turns.append(0)

        if not path.start.open_airspace and (0 not in self.os_keys_dict_pred[self.start_index_previous].keys()):    

            linestring=edges[self.start_index_previous][self.start_index].geometry
            coords = list(linestring.coords)
            for c in range(len(coords)-1):
                if (not c==0) and (lies_between(tuple((coords[c][0],coords[c][1])),tuple((self.start_point.x,self.start_point.y)),tuple((path.start.lon,path.start.lat)))):
                    tmp=(coords[c][0],coords[c][1]) #the points before the first node
                    route_centers.append(tmp) 
                    group_numbers.append(path.start.group)
                    next_node_index.append(self.start_index)
                    turns.append(0)
                

            next_node_index.append(self.start_index)
            tmp=(path.start.lon,path.start.lat)
            group_numbers.append(path.start.group)
            route_centers.append(tmp)
            turns.append(0)
        if path.start.group==-1:
            airspace_transitions.append(0)
            in_open_airspace=True
        
        group=path.start.group            

        
        selected_nodes_index=[]
        selected_nodes_index.append(path.start.index)

        while path.start.key_index!=path.goal.key_index :
            
    
            current_node=path.start
            minim=float('inf')
  
            for ch in path.start.children:
                n=graph[ch]
               
                if compute_c(path.start, n,edges,path.speed)+n.g<minim:
                    minim=compute_c(path.start, n,edges,path.speed)+n.g
                    current_node=n
                    
                    
            if current_node.index in selected_nodes_index:
                print(selected_nodes_index)
                print(current_node.index)
                print("get_path stack !! Please report this!")
                break
                
            selected_nodes_index.append(current_node.index)
            

                    
            if current_node.key_index!=path.start.key_index:
                
                if not current_node.open_airspace and not path.start.open_airspace:
                    #find the intermediate points
                    pp=1
                    linestring=edges[path.start.key_index][current_node.key_index].geometry #if the start index should go first need to get checked
                    coords = list(linestring.coords)
                    for c in range(len(coords)-1):
                        if not c==0:
                            tmp=(coords[c][0],coords[c][1]) #the intermediate point
                            route_centers.append(tmp) 
                            group_numbers.append(current_node.group)
                            next_node_index.append(current_node.key_index)
                            turns.append(0)

                if current_node.key_index!=path.goal.key_index or not current_node.open_airspace:

                    next_node_index.append(current_node.key_index)
                    tmp=(current_node.lon,current_node.lat) #the next node
                    route_centers.append(tmp)
                    
                    group_numbers.append(current_node.group)
                    turns.append(0) 

                    
                    if current_node.group==-1 and not in_open_airspace:

                        airspace_transitions.append(len(group_numbers)-1)
                        in_open_airspace=True
                    elif current_node.group!=-1 and  in_open_airspace:

                        airspace_transitions.append(len(group_numbers)-1)
                        in_open_airspace=False
                        tmp=(current_node.lon,current_node.lat) #the next node
                        route_centers.append(tmp)
                
   
            path.start=current_node
            
           
          
        if not path.goal.open_airspace: 
    
            linestring=edges[self.goal_index][self.goal_index_next].geometry
            coords = list(linestring.coords)
            for c in range(len(coords)-1):
                if (not c==0) and (lies_between(tuple((coords[c][0],coords[c][1])),tuple((self.goal_point.x,self.goal_point.y)),tuple((path.goal.lon,path.goal.lat)))):
                    tmp=(coords[c][0],coords[c][1]) #the points before the first node
                    route_centers.append(tmp) 
                    group_numbers.append(path.goal.group)
                    next_node_index.append(self.goal_index_next)
                    turns.append(0)
                

            tmp=(self.goal_point.x,self.goal_point.y)
            route_centers.append(tmp)
            group_numbers.append(path.goal.group)
            next_node_index.append(self.goal_index_next)
            turns.append(0)   
        else:
            tmp=(self.goal_point.x,self.goal_point.y)
            route_centers.append(tmp)
            group_numbers.append(path.goal.group)
            next_node_index.append(self.goal_index)
            turns.append(0)  

        if in_open_airspace:
            airspace_transitions.append(len(group_numbers)-1)
            in_open_airspace=False
        
        delete_indices=[]


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
            p2=[route_centers[airspace_transitions[open_i+1]][0],route_centers[airspace_transitions[open_i+1]][1]]
            transformer2 = Transformer.from_crs( 'epsg:4326','epsg:32633')
            
            p1=transformer2.transform(p1[1],p1[0])
            p2=transformer2.transform(p2[1],p2[0])  

            
            
            for j in range(len(next_node_index)-1): 
                
                if j+1>airspace_transitions[open_i+1]:
                    if open_i+2<len(airspace_transitions):
                        open_i=open_i+2
                    
                        p1=[route_centers[airspace_transitions[open_i]-1][0],route_centers[airspace_transitions[open_i]-1][1]]
                        p2=[route_centers[airspace_transitions[open_i+1]][0],route_centers[airspace_transitions[open_i+1]][1]]
                        transformer2 = Transformer.from_crs( 'epsg:4326','epsg:32633')
                        
                        p1=transformer2.transform(p1[1],p1[0])
                        p2=transformer2.transform(p2[1],p2[0])

    
                if  group_numbers[j]!=-1 and j!=0: 
                    route.append(route_centers[j])



                else:
                    if j==0 and  group_numbers[j]!=-1:
                        i=self.start_index_previous
                    elif group_numbers[j]!=-1:
                        i=next_node_index[j-1]
                    else:
                        i=0
    
                    ii=next_node_index[j]


                    node1=self.graph[self.os_keys_dict_pred[ii][i]]

                    if group_numbers[j]!=-1 or group_numbers[j+1]!=-1:
                        i=next_node_index[j]
                    else:
                        i=0
                    
                    ii=next_node_index[j+1]
                    
                    
                    if group_numbers[j]==-1 and group_numbers[j+1]!=-1:
                        delete_indices.append(j+1)

                    if group_numbers[j+1]!=-1 or next_node_index[j+1]==next_node_index[j]: ##That should be deleted

                        continue
                    


                    node2=self.graph[self.os_keys_dict_pred[ii][i]]
                    
                    
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
            
        for i in range(len(delete_indices)):

            
            j=delete_indices[len(delete_indices)-1-i]
            del group_numbers[j]
            del turns[j]

            

            
        ##Check for turn points
        lat_prev=self.start_point.x
        lon_prev=self.start_point.y
        
        turn_speed=copy.deepcopy(turns) ## TODO : update those values when they are ready
        #speed set to 0 for open airspace or for no turn
        #speed to 10 knots for angles smaller than 45 degrees
        #speed to 5 knots for turning angles between 45 and 90 degrees
        #speed to 2 knots for turning angles larger tha 90 degrees

###############################Retriev taht to previous state!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for i in range(len(group_numbers)-3):#for i in range(len(group_numbers)-2):
            lat_cur=route[i][0]
            lon_cur=route[i][1]
            lat_next=route[i+1][0]
            lon_next=route[i+1][1]
            ##Check the angle between the prev point- current point and the current point- next point  
            line_string_1 = [(lat_prev,lon_prev), (lat_cur,lon_cur)]
            line_string_2 = [(lat_cur,lon_cur), (lat_next,lon_next)]
            angle = 180 - angleBetweenTwoLines(line_string_1,line_string_2)

            if angle>self.cutoff_angle and turns[i]!=1 and group_numbers[i]!=-1:
                turns[i]=1
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
            if group_numbers[i]==group_numbers[i+1] or group_numbers[i+1]==-1:
                continue
            elif turns[i]!=1:
                turns[i]=1
                tmp=(route[i][1],route[i][0])
                turn_coords.append(tmp)

        turn_coords.append((-999,-999))
        turns[0]=0

        for g,i in enumerate(group_numbers):
            if i==-1 :#or ( g>0 and group_numbers[g-1]==-1)or ( g<len(group_numbers)-1 and group_numbers[g+1]==-1):
                in_constrained.append(0)
            else:
                in_constrained.append(1)        



        return route,turns,next_node_index,turn_coords,group_numbers,in_constrained,turn_speed
 
    def update_changed_vertices(self,path,graph,edges,edges_old=None,change=False,change_list=[]):
        
        if change: #Scan for changes
        ##replan
            path.k_m=path.k_m+heuristic(graph[path.origin_node_index],path.start,path.speed)
            for c in change_list:
                
                if not  c[0].expanded or not c[1].expanded:
                    print("not expanded")
                    

                c_old=compute_c(c[0], c[1],edges_old,path.speed)

                    #update cost and obstacles here
                if c_old>compute_c(c[0], c[1],edges,path.speed): #if cost is decreased

                    if(c[0]!=path.goal):
    
                        c[0].rhs=min(c[0].rhs,compute_c(c[0], c[1], edges)+c[1].g,path.speed)

                            
                elif c[0].rhs== c_old+c[1].g: #if cost is increased

                    if c[0]!=path.goal:
                        tt=[]
                        for ch in c[0].children:
                            child=graph[ch]
                            tt.append(child.g+compute_c( c[0],child, edges,path.speed))
                        c[0].rhs=min(tt)
                        path.start.rhs=float('inf')## not sure for that

                update_vertex(path, c[0])
                
                edges_old[c[0].key_index][c[1].key_index].speed=edges[c[0].key_index][c[1].key_index].speed   
 
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
        route=None
        turns=None
        groups=None
        edges_list=None
        next_turn_point=None

        self.start_point=Point(tuple((lon,lat)))
        
        edges_g=copy.deepcopy(self.edge_gdf)
        ## check for changes in the aircrafts subgraph if any
        cnt=0
        expanded=False
        change_list=[]
        
        replan_bool=True
        

        for change in changes_list:
            
            
            k=change[0]
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
            if k in self.edge_gdf.keys():
                if kk in self.edge_gdf[k].keys():
                    edges_g[k][kk].speed=change[2]
                    cnt=cnt+1
                    tmp=[]
                    ind=self.os_keys_dict_succ[k][kk]
                    if self.graph[ind].expanded:
                        expanded=True
                    tmp.append(self.graph[ind])
                    ind=self.os_keys_dict_pred[kk][k]
                    if self.graph[ind].expanded:
                       expanded=True
                    tmp.append(self.graph[ind])
                    if expanded:
                        change_list.append(tmp)
              
        if prev_node_osmnx_id!=0 and replan_bool:
            if prev_node_osmnx_id in self.edge_gdf.keys():
                if next_node_index in self.edge_gdf[prev_node_osmnx_id].keys():

                    # Do not replan in high traffic if you have low priority, should the same happen when in loitering mission?
                    if (self.edge_gdf[prev_node_osmnx_id][next_node_index].speed<1 and self.edge_gdf[prev_node_osmnx_id][next_node_index].speed!=0 and self.priority==3):# or self.edge_gdf[prev_node_osmnx_id][next_node_index].speed==0:
                        replan_bool=False
            
        if not replan_bool and change_list!=[]:
            self.update_changed_vertices(self.path,self.graph,edges_g,self.edge_gdf,True,change_list)
            self.edge_gdf=copy.deepcopy(edges_g)
            
        if cnt>0 and change_list!=[] and replan_bool:
            
            if prev_node_osmnx_id in self.os_keys_dict_pred[next_node_index].keys():
                start_id=self.os_keys_dict_pred[next_node_index][prev_node_osmnx_id]
            else:
                prev_node_osmnx_id =0
                start_id=self.os_keys_dict_pred[next_node_index][prev_node_osmnx_id]
            start_node=self.graph[start_id] 
            self.path.start=start_node
            

            self.start_index=next_node_index
            self.start_index_previous=prev_node_osmnx_id

            ##call get path
            route,turns,indices_nodes,turn_coord,groups,in_constrained,turn_speed=self.get_path(self.path,self.graph,edges_g,self.edge_gdf,True,change_list)
            self.path.origin_node_index=start_id
             
            if route != None :
                edges_list=[]
                next_turn_point=[]
                os_id1=self.start_index_previous

                os_id2=indices_nodes[0]
                cnt=0
                for i in range(0,len(indices_nodes)-1):
                    
                        
                    if indices_nodes[i]==-1 or indices_nodes[i]==os_id2:
                        cnt=cnt+1
                    else:
                        for j in range(cnt):
                            edges_list.append((os_id1,os_id2))
                        #edges_list.append((os_id1,indices_nodes[i]))
                        cnt=1
                        os_id1=os_id2
                        os_id2=indices_nodes[i]
                        if i>0 and groups[i]==-1 and groups[i-1]==-1:
                            os_id1=0
                            
                    if i>0 and groups[i]!=-1 and groups[i-1]==-1:
                        os_id2=indices_nodes[i]
                        del edges_list[-1]
                        #continue
                for j in range(cnt):
                    edges_list.append((os_id1,os_id2))
                
                if self.goal_index_next!=0:
                    edges_list.append((self.goal_index,self.goal_index_next))
                else:
                    edges_list.append((0,self.goal_index))

                            
                cnt=0
                for i in turns:
                    if i:
                        next_turn_point.append(turn_coord[cnt])
                        cnt=cnt+1
                    else:
                        next_turn_point.append(turn_coord[cnt])
                                
                del indices_nodes[0]
                        
                self.edge_gdf=copy.deepcopy(edges_g)
                            
                self.route=route
                self.turns=turns
                self.edges_list=edges_list
                self.next_turn_point=next_turn_point
                self.groups=groups  
                self.in_constrained=in_constrained
                self.turn_speed=turn_speed
                return self.route,self.turns,self.edges_list,self.next_turn_point,self.groups,self.in_constrained,self.turn_speed
            
        elif cnt>0:
            self.edge_gdf=copy.deepcopy(edges_g)
            

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

        route=None
        turns=None
        groups=None
        edges_list=None
        next_turn_point=None

        
        self.start_point=Point(tuple((lon,lat)))
        
        edges_g=copy.deepcopy(self.edge_gdf)
        ## check for changes in the aircrafts subgraph if any
        cnt=0
        expanded=False
        change_list=[]
        
        replan_bool=True
        


        for change in changes_list:

            k=change[0]
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
            if k in self.edge_gdf.keys():
                if kk in self.edge_gdf[k].keys():
                    edges_g[k][kk].speed=change[2]
                    cnt=cnt+1
                    tmp=[]
                    ind=self.os_keys_dict_succ[k][kk]
                    if self.graph[ind].expanded:
                        expanded=True
                    tmp.append(self.graph[ind])
                    ind=self.os_keys_dict_pred[kk][k]
                    if self.graph[ind].expanded:
                       expanded=True
                    tmp.append(self.graph[ind])
                    if expanded:
                        change_list.append(tmp)
              
        if prev_node_osmnx_id!=0 and replan_bool:
            if prev_node_osmnx_id in self.edge_gdf.keys():
                if next_node_index in self.edge_gdf[prev_node_osmnx_id].keys():

                    # Do not replan in high traffic if you have low priority, should the same happen when in loitering mission?
                    if (self.edge_gdf[prev_node_osmnx_id][next_node_index].speed<1 and self.edge_gdf[prev_node_osmnx_id][next_node_index].speed!=0 and self.priority==3):# or self.edge_gdf[prev_node_osmnx_id][next_node_index].speed==0:
                        replan_bool=False

            
        if not replan_bool and change_list!=[]:
            self.update_changed_vertices(self.path,self.graph,edges_g,self.edge_gdf,True,change_list)
            self.edge_gdf=copy.deepcopy(edges_g)
            
        if cnt>0 and change_list!=[] and replan_bool:
            
            if prev_node_osmnx_id in self.os_keys_dict_pred[next_node_index].keys():
                start_id=self.os_keys_dict_pred[next_node_index][prev_node_osmnx_id]
            else:
                prev_node_osmnx_id =0
                start_id=self.os_keys_dict_pred[next_node_index][prev_node_osmnx_id]
            start_node=self.graph[start_id] 
            self.path.start=start_node
            
# =============================================================================
#             self.start_index=next_node_index
#             self.start_index_previous=prev_node_osmnx_id
# =============================================================================


            ##call get path
            route,turns,indices_nodes,turn_coord,groups,in_constrained,turn_speed=self.get_path(self.path,self.graph,edges_g,self.edge_gdf,True,change_list)
            self.path.origin_node_index=start_id
             
            if route != None :
                edges_list=[]
                next_turn_point=[]
                os_id1=self.start_index_previous

                os_id2=indices_nodes[0]
                cnt=0
                for i in range(0,len(indices_nodes)-1):
                    
                        
                    if indices_nodes[i]==-1 or indices_nodes[i]==os_id2:
                        cnt=cnt+1
                    else:
                        for j in range(cnt):
                            edges_list.append((os_id1,os_id2))
                        #edges_list.append((os_id1,indices_nodes[i]))
                        cnt=1
                        os_id1=os_id2
                        os_id2=indices_nodes[i]
                        if i>0 and groups[i]==-1 and groups[i-1]==-1:
                            os_id1=0
                            
                    if i>0 and groups[i]!=-1 and groups[i-1]==-1:
                        os_id2=indices_nodes[i]
                        del edges_list[-1]
                        #continue
                for j in range(cnt):
                    edges_list.append((os_id1,os_id2))
                    
                if self.goal_index_next!=0:
                    edges_list.append((self.goal_index,self.goal_index_next))
                else:
                    edges_list.append((0,self.goal_index))
    
                            
                cnt=0
                for i in turns:
                    if i:
                        next_turn_point.append(turn_coord[cnt])
                        cnt=cnt+1
                    else:
                        next_turn_point.append(turn_coord[cnt])
                                
                del indices_nodes[0]
                        
                self.edge_gdf=copy.deepcopy(edges_g)
                            
                self.route=route
                self.turns=turns
                self.edges_list=edges_list
                self.next_turn_point=next_turn_point
                self.groups=groups  
                self.in_constrained=in_constrained
                self.turn_speed=turn_speed
                return self.route,self.turns,self.edges_list,self.next_turn_point,self.groups,self.in_constrained,self.turn_speed
            
        elif cnt>0:
            self.edge_gdf=copy.deepcopy(edges_g)
            

        return self.route,self.turns,self.edges_list,self.next_turn_point,self.groups,self.in_constrained,self.turn_speed