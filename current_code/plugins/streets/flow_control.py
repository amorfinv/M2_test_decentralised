# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:07:51 2021

@author: nipat
# kdtree code from https://github.com/chuducty/KD-Tree-Python
"""

import heapq
import math
import numpy as np
import random


##https://github.com/chuducty/KD-Tree-Python

def kselect(li,k):
    
    while True:
        r = random.randint(0,len(li)-1) # random a number
        cur = li[r]
        #swap to the first element
        tmp = li[0]
        li[0] = li[r]
        li[r] = tmp

        # partion around the pivot, to find a rank of the chosen element
        #-------------
        j = 1
        
        for i in range(1,len(li)):
            if li[i] <= cur:
                # swap with the j
                tmp = li[j]
                li[j] = li[i]
                li[i] = tmp
                j += 1
            
        j -= 1
        tmp = li[0]
        li[0] = li[j]
        li[j] = tmp
        #--------------
        cur_rank = j + 1
        if k == cur_rank:
            return li[k-1]
        #if the random element have a rank not in range [n/3,2n/3], we redo everything
        if cur_rank > len(li) / 3 or cur_rank < 2*len(li)/3:
            break
    #print(cur,cur_rank,li)
    
    if k < cur_rank:
        return kselect(li[0:cur_rank-1],k)
    else:
        return kselect(li[cur_rank:],k-cur_rank)
    

class Node:
    def __init__(self,key_index,x,y):
        self.key_index=key_index # the index the osmnx graph
        self.x=x
        self.y=y
    
        self.parents=[]
        self.children=[]
        

class Edge:
    av_speed_horizontal=0.005#10.0
    av_speed_vertical=2.0
    def __init__(self,start,end,length,geometry,stroke):
        self.start_key=start #the osmnx key of the start node
        self.end_key=end #the osmnx key of the end node
        self.length=length
        self.geometry=geometry
        self.max_speed=10 #max allowed overall speed
        self.speed=10 #max speed allowed after geovectoring
        self.max_density=1#the maximum allowed density
        self.density=0#the measured density
        self.stroke_group=stroke
        self.layer_alt=25# the altitude of the lower layer allowed in that edge in feet
            
class bbox:
    def __init__(self,x1,y1,x2,y2):
        self.x1=x1
        self.y1=y1
        self.x2=x2
        self.y2=y2
        
class Tree_node:
    def __init__(self,x,y,os_key):
        self.x = x
        self.y = y
        self.os_key=os_key
        self._key = None
        self._line = None
        self._l = None
        self._r = None
        self._P = None
        #(xmin,xmax,ymin,ymax)
        self._area = (-math.inf,math.inf,-math.inf,math.inf)


        
class KdTree(object):
    def __init__(self):
        self._root = Tree_node(None,None,None)
        universe_node =Tree_node(None,None,None)
        
        self._root._p = universe_node
        self._range_query = []
        self._leaf = 0

    def build(self,li):
        self._root = self.build_recur(li,0)
    def build_recur(self,li,depth):
        
        if len(li) == 1:
            node = Tree_node(li[0][0],li[0][1],li[0][2])
            node._key = li[0][0],li[0][1]
            return node
        li_tmp = []
        
        for i in li:
            li_tmp.append(i[depth % 2])               
        middle = kselect(li_tmp,len(li)//2+1)
    
        
        left_li = []
        right_li = []
 
        for i in li:
            if not (i[2]==li[0][2]):
                if i[depth % 2] >= middle:
                    right_li.append(i)
                if i[depth % 2] < middle:
                    left_li.append(i)
        middle_node = Tree_node(li[0][0],li[0][1],li[0][2])
        middle_node._key= li[0][0],li[0][1]
        middle_node._line = middle
        
        if left_li != []:
            middle_node._l = self.build_recur(left_li,depth + 1)
        if right_li != []:
            middle_node._r = self.build_recur(right_li,depth + 1)
        if depth % 2 == 0:
            xmin,xmax,ymin,ymax = middle_node._area
            if left_li != []:
                middle_node._l._area = (xmin,middle,ymin,ymax)
            if right_li != []:
                middle_node._r._area = (middle,xmax,ymin,ymax)
        else:
            xmin,xmax,ymin,ymax = middle_node._area
            if left_li != []:
                middle_node._l._area = (xmin,xmax,ymin,middle)
            if right_li != []:
                middle_node._r._area = (xmin,xmax,middle,ymax)
        return middle_node

    def intersect(self,a1,a2):
        x1min,x1max,y1min,y1max = a1
        x2min,x2max,y2min,y2max = a2
        if (x1max < x2min or x2max < x1min or y1max < y2min or y2max < y1min):
            return False
        return True
    
    def query(self,area):
        self._range_query = []
        self.query_recur(self._root,area)
        return self._range_query
    def query_recur(self,node,area):
        if node == None:
            return
        if node._line == None: # if it's a leaf node a.k.a a point
            if node._key == None:
                return
            #x,y = node._key
            x=node.x
            y=node.y
            xmin,xmax,ymin,ymax = area
            if (x >= xmin and x <= xmax and y >= ymin and y <= ymax):
                self._range_query.append(node.os_key)
            return
        else:
            if node._key == None:
                return
            #x,y = node._key
            x=node.x
            y=node.y
            xmin,xmax,ymin,ymax = area
            if (x >= xmin and x <= xmax and y >= ymin and y <= ymax):
                self._range_query.append(node.os_key)
        
        if node._l:
            if self.intersect(area,node._l._area):
                self.query_recur(node._l,area)
        if node._r:
            if self.intersect(area,node._r._area):
                self.query_recur(node._r,area)
        
        ##self.query_recur(node._l,area)
        ##self.query_recur(node._r,area)

    ##Compute the eucledean distance between two points
    def distance_eucledian(self,p1,p2):
        d=math.sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)) 
        return d
    
    ##Compute the distance between two geodetic points
    def distance(self,p1,p2): ##harvestine distance
        R = 6372800  # Earth radius in meters
        lat1=p1.y
        lon1=p1.x
        lat2=p2.y
        lon2=p2.x 
        
        phi1, phi2 = math.radians(lat1), math.radians(lat2) 
        dphi       = math.radians(lat2 - lat1)
        dlambda    = math.radians(lon2 - lon1)
        
        a = math.sin(dphi/2)**2 + \
            math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        
        return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))   
    
    # For the closest neighbor
    def get_nearest(self,kd_node, point, return_distances=True, i=0, best=None):
        if kd_node is not None:
            dist = self.distance(point, kd_node)
            if i==1:
                dx = kd_node.y - point.y
            else:
                dx = kd_node.x - point.x
            if not best:
                best = [dist, kd_node.os_key]
            elif dist < best[0]:
                best[0], best[1] = dist, kd_node.os_key
            i = (i + 1) % 2
            # Goes into the left branch, and then the right branch if needed
            #for b in [dx < 0] + [dx >= 0] * (dx * dx < best[0]):
            #if dx<0:
            #    self.get_nearest(kd_node._l, point, return_distances, i, best)
                
            #if [dx >= 0] *(dx * dx < best[0]):
            #    self.get_nearest(kd_node._r, point, return_distances, i, best)
                
            if dx<0:
                self.get_nearest(kd_node._l, point, return_distances, i, best)
                
            else:
                self.get_nearest(kd_node._r, point, return_distances, i, best)
            if (dx * dx < best[0]):
                if dx<0:
                    self.get_nearest(kd_node._r, point, return_distances, i, best)
                else:
                    self.get_nearest(kd_node._l, point, return_distances, i, best)
                    

        return best if return_distances else best[1]
            
            
    def count_leaf(self):
        self._leaf = 0
        self.count(self._root)
        return self._leaf
    def count(self,node):
        if node == None:
            return
        if node._key != None:
            self._leaf +=1
        
        self.count(node._l)
        self.count(node._r)


            
class street_graph:                
    def __init__(self,G,edges_gdf):
        self.nodes_graph={}
        self.edges_graph={}
        self.modified={}
        self.create_graph(G,edges_gdf)
        self.create_tree(G)
        self.G=G

        
    ##Create a kd tree with the flow control nodes
    def create_tree(self,G):
        tree_nodes = []
        self.kdtree= KdTree()
        omsnx_keys_list=list(G._node.keys())
        for key in omsnx_keys_list:
            tmp=[self.nodes_graph[key].x,self.nodes_graph[key].y,key]
            tree_nodes.append(tmp)
        self.kdtree.build(tree_nodes)
        
    ##Create the flow control graph 
    def create_graph(self,G,edges_gdf):
        #edges_gdf=pickle.load(open("edge_gdf.pickle", "rb"))#load edge_geometry
        omsnx_keys_list=list(G._node.keys())
        G_list=list(G._node)
        for key in omsnx_keys_list:
            x=G._node[key]['x']
            y=G._node[key]['y']
            node=Node(key,x,y)
            children=list(G._succ[key].keys())
            for ch in children: 
                node.children.append(ch)#node.children.append(G_list.index(ch))
                length=G[key][ch][0]['length']
                geom=G[key][ch][0]['geometry']#edges_geometry[key][ch][0]
                stroke_group=edges_gdf.loc[key].loc[ch].loc[0]['stroke_group']
                edge=Edge(key,ch,length,geom,stroke_group)
                tmp={}
                tmp[ch]=edge
                if key in self.edges_graph.keys():
                    self.edges_graph[key][ch]=edge
                    self.modified[key][ch]=0
                else:
                    self.edges_graph[key]=tmp
                    tt={}
                    tt[ch]=0
                    self.modified[key]=tt
            parents=list(G._pred[key].keys())
            for p in parents:
                node.parents.append(p)#node.parents.append(G_list.index(p))
                length=G[p][key][0]['length']
                geom=G[p][key][0]['geometry']#geom=edges_geometry[p][key][0]
                stroke_group=edges_gdf.loc[p].loc[key].loc[0]['stroke_group']
                edge=Edge(p,key,length,geom,stroke_group)
                tmp={}
                tmp[key]=edge
                if p in self.edges_graph.keys():
                    self.edges_graph[p][key]=edge
                    self.modified[p][key]=0
                else:
                    self.edges_graph[p]=tmp
                    tt={}
                    tt[key]=0
                    self.modified[p]=tt
            self.nodes_graph[key]=node
        

    ##Get the nearest node of a point (x,y) 
    def get_nearest_node(self,x,y):
        
        point=Tree_node(x,y,-1)
        distance, index = self.kdtree.get_nearest(self.kdtree._root,point)
        return distance,index
    
    ##Get the nodes that lie in an area box
    def get_nodes_in_area(self,box):
        
        area=(box.y1,box.y2,box.x1,box.x2)
        keys =self.kdtree.query(area)

        return keys
     
    #Get the graph of nodes and edges that lie in an area box
    def extract_subgraph(self,box):
        graph={}
        edges={}
        
        keys=self.get_nodes_in_area(box)

        for i in keys:
            graph[i]=self.nodes_graph[i]
           
    
        for i in keys:
            tmp_ch=[]
            for ch in graph[i].children:
                if ch in keys:
                    tmp={}
                    tmp[ch]=self.edges_graph[i][ch]
                    if i in edges.keys():
                        edges[i][ch]=self.edges_graph[i][ch]
                    else:
                        edges[i]=tmp
                else:
                    tmp_ch.append(ch)

            for ch in tmp_ch:
                graph[i].children.remove(ch)

            tmp_p=[]       
            for p in graph[i].parents:
                if p in keys:
                    tmp={}
                    tmp[i]=self.edges_graph[p][i]
                    if p in edges.keys():
                        edges[p][i]=self.edges_graph[p][i]
                    else:
                        edges[p]=tmp
                else:
                    tmp_p.append(p)

            for p in tmp_p:
                graph[i].parents.remove(p)
                    
        return graph,edges
    

    
    def compute_density(edge):
        
        density=0
        
    def update_geovectors(self):
        ##Compute the densities
        ##After computing a density, if teh density is high the density of the parent edge should be assigned a high value as well
        
        edges_changes=[]
        
        edge_keys=self.edges_graph.keys()
        
        for i in edge_keys:
            keys=self.edges_graph[i].keys()
            for j in keys:
                edge=self.edge_graph[i][j]
                if edge.density>=0.5* edge.max_density:
                    if edge.speed!=edge.max_speed/2:
                        tmp=[i,j,edge.max_speed/2]#the keys of the vertices of the edges, followed by the new speed
                        edges_changes.append(tmp)
                    edge.speed=edge.max_speed/2
                elif edge.density<0.5* edge.max_density:
                    if edge.speed!=edge.max_speed:
                        tmp=[i,j,edge.max_speed]
                        edges_changes.append(tmp)
                    edge.speed=edge.max_speed
                    
        return edges_changes
        
    def add_no_fly_zone():
        speed=0
        
