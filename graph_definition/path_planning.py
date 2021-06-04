# -*- coding: utf-8 -*-
"""
Created on Wed May  5 12:19:13 2021

@author: nipat
"""
import osmnx
import matplotlib.pyplot as plt
import heapq
import pandas as pd
import numpy as np
import os

class Node:
    av_speed_horizontal=0.005#10.0
    av_speed_vertical=2.0
    def __init__(self,key_index,x,y,z,index,group):
        self.key_index=key_index # the index the osmnx graph
        self.index=index# the index in the search graph

        #the coordinates of the node as given by osmnx (latitude,longitude)
        self.x=x
        self.y=y
        self.z=z

        #the parents(predessecors) and children(successor) of the node expressed as lists containing their indexes in the graph 
        self.parents=[]
        self.children=[]
        
        #self.f=0.0
        self.g=float('inf')
        self.rhs=float('inf')
        self.h=0.0
        self.key=[0.0,0.0]

        self.density=1.0 #shows the traffic density      

        self.inQueue=False
        
        #the travelling speed
        #that will probably depend on the group
        self.speed=13
        #the stroke group
        self.group=group

        
class Path:
    def __init__(self,start,goal):
        self.start=start
        self.goal=goal
        self.k_m=0
        self.queue=[]
        
        
def initialise(path):
    path.queue=[]
    path.k_m=0
    path.goal.rhs=0
    path.goal.inQueue=True
    path.goal.h=heuristic(path.start,path.goal)
    heapq.heappush(path.queue, (path.goal.h,0,path.goal.x,path.goal.y,path.goal.z,path.goal.index, path.goal))
 


def compare_keys(node1,node2):
    if node1[0]<node2[0]:
        return True
    elif node1[0]==node2[0] and node1[1]<node2[1]:
        return True
    return False
        
def calculateKey(node,start, k_m):
    return (min(node.g, node.rhs) + heuristic(node,start) + k_m, min(node.g, node.rhs))

def heuristic(current, goal):
    h=abs(goal.x-current.x)/current.av_speed_horizontal+abs(goal.y-current.y)/current.av_speed_horizontal+abs(goal.z-current.z)/current.av_speed_vertical
    return h

def compute_c(current,neigh, G):
    g=1
    if(current.z!=neigh.z):
        g=abs(neigh.z-current.z)/current.av_speed_vertical
    else:
        #g=(abs(neigh.x-current.x)+abs(neigh.y-current.y))*2/(current.speed/current.density+neigh.speed/neigh.density)
        #here i need to check if the group is changing (the drone needs to turn)
        if current.group==neigh.group:
            g=G[current.key_index][neigh.key_index][0]['length']/current.speed

        else:
            g=1 #set a standard cost for going to the turn layer
    return g

def top_key(q):
    if len(q) > 0:
        return [q[0][0],q[0][1]]
    else:
        # print('empty queue!')
        return [float('inf'), float('inf')]
    
def update_vertex(path,node):
    if node.g!=node.rhs and node.inQueue:
        #Update
        node.key=calculateKey(node, path.start, path.k_m)
        id_in_queue = [item for item in path.queue if node in item]
        if id_in_queue != []:
            if len(id_in_queue) != 1:
                raise ValueError('more than one ' + node + ' in the queue!')
            path.queue.remove(id_in_queue[0])
            heapq.heappush(path.queue, (node.key[0],node.key[1],node.x,node.y,node.z,node.index,node))
    elif node.g!=node.rhs and not node.inQueue:
        #Insert
        node.inQueue=True
        node.h=heuristic(node, path.start)
        node.key=calculateKey(node, path.start, path.k_m)
        heapq.heappush(path.queue, (node.key[0],node.key[1],node.x,node.y,node.z,node.index,node))
    elif node.g==node.rhs and node.inQueue:
        #remove
        node.inQueue=False
        id_in_queue = [item for item in path.queue if id in item]
        if id_in_queue != []:
            if len(id_in_queue) != 1:
                raise ValueError('more than one ' + id + ' in the queue!')
            path.queue.remove(id_in_queue[0])
            
def compute_shortest_path(path,graph, G):
    path.start.key=calculateKey(path.start, path.start, path.k_m)
    k_old=top_key(path.queue)
   
    while path.start.rhs>path.start.g or compare_keys(k_old,path.start.key):
        if len(path.queue)==0:
            print("No path found!")
            return 0
        #print(len(path.queue))
        k_old=top_key(path.queue)
        current_node=path.queue[0][6]#get the node with teh smallest priority
        k_new=calculateKey(current_node, path.start, path.k_m)
        if compare_keys(k_old, k_new):
            current_node.key=k_new
            id_in_queue = [item for item in path.queue if current_node is item]
            if id_in_queue != []:
                if len(id_in_queue) != 1:
                    raise ValueError('more than one ' + current_node + ' in the queue!')
                path.queue.remove(id_in_queue[0])
                heapq.heappush(path.queue, (current_node.key[0],current_node.key[1],current_node.x,current_node.y,current_node.z,current_node.index,current_node))
        elif current_node.g>current_node.rhs:
            current_node.g=current_node.rhs
            id_in_queue = [item for item in path.queue if current_node in item]
            if id_in_queue != []:
                if len(id_in_queue) != 1:
                    raise ValueError('more than one ' + current_node + ' in the queue!')
                path.queue.remove(id_in_queue[0])
                current_node.inQueue=False

                for p in current_node.parents:
                    pred_node=graph[p]  
                    if pred_node!=path.goal:
                        pred_node.rhs=min(pred_node.rhs,current_node.g+compute_c(pred_node,current_node, G))
                    update_vertex(path, pred_node)
        else:
            g_old=current_node.g
            current_node.g=float('inf')
            pred_node=current_node
            if pred_node.rhs==g_old:
                if pred_node!= path.goal:
                    tt=()
                    for ch in pred_node.children:
                        child=graph[ch]
                        tt.append(child.g+compute_c(child, pred_node, G))
                    pred_node.rhs=min(tt)
            for p in current_node.parents:
                parent=graph[p]
                if parent.rhs==(g_old+compute_c(current_node, parent, G)):
                    if(parent!=path.goal):
                        tt=()
                        for ch in parent.children:
                            child=graph[ch]
                            tt.append(child.g+compute_c(child, parent, G))
                        parent.rhs=min(tt)
                update_vertex(path, parent)
            pred_node=current_node
            if pred_node.rhs==g_old:
                if pred_node!= path.goal:
                    tt=()
                    for ch in pred_node.children:
                        child=graph[ch]
                        tt.append(child.g+compute_c(child, pred_node, G))
                    pred_node.rhs=min(tt)
            update_vertex(path, pred_node)
    return 'Path found!'
            
            
def get_path(path,graph, G):
    change=False
    edges_geometry=osmnx.graph_to_gdfs(G, nodes=False)["geometry"]
    change_list=[]# a list with the nodes between which the cost changed
    route=[]
    group_numbers=[]
    tmp=(path.start.x,path.start.y,path.start.z)
    route.append(tmp)
    group=path.start.group
    while path.start.key_index!=path.goal.key_index :
        
        current_node=path.start
        minim=float('inf')
        for ch in path.start.children:
            n=graph[ch]
            if compute_c(path.start, n, G)+n.g<minim:
                minim=compute_c(path.start, n, G)+n.g
                current_node=n

        if current_node.key_index!=path.start.key_index:
            #find the intermediate points
            pp=1
            linestring=edges_geometry[path.start.key_index][current_node.key_index][0] #if the start index should go first need to get checked
            coords = list(linestring.coords)
            for c in range(len(coords)-1):
                if not c==0:
                    tmp=(coords[c][0],coords[c][1],current_node.z) #the intermediate point
                    route.append(tmp) 
                    group_numbers.append(current_node.group)
                

            tmp=(current_node.x,current_node.y,current_node.z) #the next node
            group_numbers.append(current_node.group)
            route.append(tmp) 
        if change: #Scan for changes
            path.k_m=path.k_m+heuristic(current_node, path.start)
            for c in change_list:
                c_old=compute_c(c[0], c[1], G)
                #update cost and obstacles here
                if c_old>compute_c(c[0], c[1]):
                    if(c[0]!=path.goal):
                        c[0].rhs=min(c[0].rhs,compute_c(c[0], c[1], G)+c[1].g)
                elif c[0].rhs== c_old+c[1].g:
                    if c[0]!=path.goal:
                        #child_list=get_child_list(path, c[0])
                        tt=[]
                        for ch in c[0].children:
                            child=graph[ch]
                            tt.append(child.g+compute_c(child, c[0], G))
                        c[0].rhs=min(tt)
                update_vertex(path, c[0])
                compute_shortest_path(path)
                
        path.start=current_node
        
        
    turns=[0]
    for i in range(len(group_numbers)-1):
        if group_numbers[i]==group_numbers[i+1]:
            turns.append(0)
        else:
            turns.append(1)
    turns.append(0)
        
    return route,turns
        
class PathPlanning:
    
    def __init__(self,G ,start_x,start_y,goal_x,goal_y):
        self.s_x=start_x
        self.s_y=start_y
        self.g_x=goal_x
        self.g_y=goal_y
        self.G = G

        osmnx.config(use_cache=True, log_console=True)
        

        # convert graph to geodataframe
        g = osmnx.graph_to_gdfs(self.G)
        
        # # get node and edge geodataframe
        nodes_gdf = g[0]
        edge_gdf = g[1]
        
        
        north_south_list=[2,5,6,7,8,9,10,11,13,14,17,19,20,21,22,23,26,25,28,29,30,31,34,36,37]#fly at 10 meters
        east_west_list=[0,1,3,4,12,15,16,18,24,27,32,33,35] #fly at 20 meters

        

        ##Create the search graph
        stroke_groups = list(np.unique(np.array(edge_gdf['stroke_group'].values)))
        edge_keys = list(self.G.edges())[0][0]
        
        #Create the graph
        self.graph=[]
        omsnx_keys_list=list(self.G._node.keys())
        G_list=list(self.G._node)
        
        new_nodes_counter=0
        for i in range(len(omsnx_keys_list)):
           key=omsnx_keys_list[i]
           x=self.G._node[key]['x']
           y=self.G._node[key]['y']  
           z = 25 # ft
           parents=list(self.G._pred[key].keys())
           children=list(self.G._succ[key].keys())
           my_group={}
        
           ii=0
           tmp=[]#list if the groups that the node has been added
           for p in parents:
               if not ii:
                   if p in edge_gdf.index and key in edge_gdf.loc[p].index: 
                       group=edge_gdf.loc[p].loc[key].loc[0]['stroke_group']
                       if int(group) in north_south_list:
                           z=25 #ft
                       elif (int(group) in east_west_list):
                           z=50 #ft
                       node=Node(key,x,y,z,i+new_nodes_counter,group)
                       my_group.update({i+new_nodes_counter:group})
                       self.graph.append(node)
                       tmp.append(group)
                       ii=ii+1
               else: 
                if p in edge_gdf.index and key in edge_gdf.loc[p].index: 
                        new_nodes_counter=new_nodes_counter+1
                        group=edge_gdf.loc[p].loc[key].loc[0]['stroke_group']
                        if int(group) in north_south_list:
                           z=25 #ft
                        elif int(group) in east_west_list:
                           z=50 #ft
                        node=Node(key,x,y,z,i+new_nodes_counter,group)
                        my_group.update({i+new_nodes_counter:group})
                        self.graph.append(node)
                        tmp.append(group)
        
                        ii=ii+1
        
           for ch in children:
                group=edge_gdf.loc[key].loc[ch].loc[0]['stroke_group']
                if not group in tmp:
                    new_nodes_counter=new_nodes_counter+1
                    if int(group) in north_south_list:
                           z=10
                    elif int(group) in east_west_list:
                           z=15
                    node=Node(key,x,y,z,i+new_nodes_counter,group)
                    my_group.update({i+new_nodes_counter:group})
                    self.graph.append(node)
                    tmp.append(group)
                    ii=ii+1
                    
        
           if ii==0:
                node=Node(key,x,y,i+new_nodes_counter,-1)
                self.graph.append(node)
                        
        
        
           if len(my_group)>1:
                for index in my_group:
                    for index_ in my_group:
                        if my_group[index]!=my_group[index_] and index!=index_:
                            self.graph[index].children.append(index_)
                            self.graph[index].parents.append(index_)
                      
        #add the children and parents to each node                
        for i in self.graph:
            key=i.key_index
            parents=list(self.G._pred[key].keys())
            children=list(self.G._succ[key].keys())
            for p in parents:
                for j in self.graph:
                    if p==j.key_index and (j.group==i.group or i.group==-1) :
                        i.parents.append(j.index)
                       
                        break
            for ch in children:
                for j in self.graph:
                    if ch==j.key_index and (j.group==i.group or i.group==-1):
                        i.children.append(j.index)
                        break
        
            

                    
    def plan(self):
                
        start_id=-1
        goal_id=-1
        #key=G_list[start_id]

        start_index=osmnx.distance.nearest_nodes(self.G,self.s_x,self.s_y, return_dist=False)
        goal_index=osmnx.distance.nearest_nodes(self.G,self.g_x,self.g_y, return_dist=False)
        
        for i in self.graph:
            if i.key_index==start_index:
                start_id=i.index
            if i.key_index==goal_index:
                goal_id=i.index
            if start_id!=-1 and goal_id!=-1:
                break
        
        start_node=self.graph[start_id] 
        x_start=start_node.x
        y_start=start_node.y#G._node[key]['y']
        
        goal_node=self.graph[goal_id] 
        
        x_goal=goal_node.x
        y_goal=goal_node.y
         
        
        
        path=Path(start_node,goal_node)
        
        initialise(path)
        
        path_found=compute_shortest_path(path,self.graph, self.G)
        print(path_found)
        
        route=[]
        turns=[]
        if path_found:
            route,turns=get_path(path,self.graph, self.G)
            
        return route,turns

# =======
# ########################
# G = osmnx.io.load_graphml(filepath='C:/Users/nipat/Downloads/M2_test_scenario-main/M2_test_scenario-main/graph_definition/gis/data/street_graph/processed_graph1.graphml')
# #provide the start and destination coordinates 
# start_x=16.3281
# start_y=48.223
# goal_x=16.34
# goal_y=48.225
# plan1=PathPlanning(start_x,start_y,goal_x,goal_y)
# =======
# x_list=[]
# y_list=[]
# x_list_up=[]
# y_list_up=[]
# for r in route:
#     if(r[2]==10):
#         x_list.append(r[0])
#         y_list.append(r[1])
#     else:
#         x_list_up.append(r[0])
#         y_list_up.append(r[1])
# ax.scatter(x_list,y_list, color='g')
# ax.scatter(x_list_up,y_list_up, color='y')
# ax.scatter(start_x,start_y, color='b')
# ax.scatter(goal_x,goal_y, color='r')
# plt.show()
