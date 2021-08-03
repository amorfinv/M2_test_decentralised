# -*- coding: utf-8 -*-
"""
Created on Wed May  5 12:19:13 2021

@author: nipat
## https://github.com/mdeyo/d-star-lite"""

# https://stackoverflow.com/questions/61467693/d-lite-search-algorithm-for-robot-path-planning-gets-stuck-in-infinite-loop-wh
# https://stackoverflow.com/questions/7245238/pathfinding-algorithm-creating-loops?rq=1
# https://stackoverflow.com/questions/61462672/ros-move-base-d-dstar-pathplanning-algorithm-implementation

import heapq
import numpy as np
import math
import copy
from flow_control import street_graph,bbox

class Node:
    av_speed_horizontal= 10.0
    av_speed_vertical=1.0
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

def distance(p1,p2): ##harvestine distance
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
    
def heuristic(current, goal):
    n1=Node(current.key_index,current.x,current.y,current.z,current.index,current.group)
    n2=Node(goal.key_index,goal.x,goal.y,goal.z,goal.index,goal.group)
    n1.y=0
    n2.y=0
    d1=distance(current,goal)

    n1.y=current.y
    n1.x=0
    n2.y=goal.y
    n2.x=0
    d2=distance(current,goal)
    h=(d1+d2)/current.av_speed_horizontal
    if current.z!=goal.z:
        h=h+5 #set a standard cost for going to the turn layer # g=5/current.av_speed_vertical
    return h

def compute_c(current,neigh, edges):
    g=1
    if(current.z!=neigh.z):
        g=abs(neigh.z-current.z)/current.av_speed_vertical
    else:
        #g=(abs(neigh.x-current.x)+abs(neigh.y-current.y))*2/(current.speed/current.density+neigh.speed/neigh.density)
        #here i need to check if the group is changing (the drone needs to turn)
        if current.group==neigh.group:
            if not edges[current.key_index][neigh.key_index].max_speed:
                g=float('inf')

            else:
                g=edges[current.key_index][neigh.key_index].length/edges[current.key_index][neigh.key_index].max_speed

        else:
            g=5 #set a standard cost for going to the turn layer # g=5/current.av_speed_vertical
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
            
def compute_shortest_path(path,graph, G,edges):
    path.start.key=calculateKey(path.start, path.start, path.k_m)
    k_old=top_key(path.queue)
   
    while path.start.rhs>path.start.g or compare_keys(k_old,path.start.key):
        if len(path.queue)==0:
            print("No path found!")
            return 0

        k_old=top_key(path.queue)
        current_node=path.queue[0][6]#get the node with the smallest priority
        
        k_new=calculateKey(current_node, path.start, path.k_m)
        if compare_keys(k_old, k_new):
            current_node.key=k_new

            #id_in_queue = [item1 for item in path.queue if current_node.index==item[6].index]
            #print(id_in_queue)
            if 1:#id_in_queue !=[] :

                #if len(id_in_queue) != 1:
                #    raise ValueError('more than one ' + current_node + ' in the queue!')
                path.queue.remove(path.queue[0])
                heapq.heappush(path.queue, (current_node.key[0],current_node.key[1],current_node.x,current_node.y,current_node.z,current_node.index,current_node))
        elif current_node.g>current_node.rhs:
            current_node.g=current_node.rhs
            
            #id_in_queue = [item for item in path.queue if current_node in item]
            if 1:#id_in_queue != []:
                #if len(id_in_queue) != 1:
                #    raise ValueError('more than one ' + current_node + ' in the queue!')
                path.queue.remove(path.queue[0])#id_in_queue[0])
                current_node.inQueue=False

                for p in current_node.parents:
                    pred_node=graph[p]  

                    if pred_node!=path.goal:
                        pred_node.rhs=min(pred_node.rhs,current_node.g+compute_c(pred_node,current_node,edges))
                    update_vertex(path, pred_node)
        else:
            g_old=current_node.g
            current_node.g=float('inf')
            pred_node=current_node
            if pred_node.rhs==g_old:
                if pred_node!= path.goal:
                    tt=[]
                    for ch in pred_node.children:
                        child=graph[ch]
                        tt.append(child.g+compute_c(child, pred_node,edges))
                    pred_node.rhs=min(tt)
            for p in current_node.parents:
                parent=graph[p]
                if parent.rhs==(g_old+compute_c(parent,current_node,edges)):
                    if(parent!=path.goal):
                        tt=[]
                        for ch in parent.children:
                            child=graph[ch]
                            tt.append(child.g+compute_c(parent,child,edges))
                        parent.rhs=min(tt)
                update_vertex(path, parent)
            pred_node=current_node
            if pred_node.rhs==g_old:
                if pred_node!= path.goal:
                    tt=()
                    for ch in pred_node.children:
                        child=graph[ch]
                        tt.append(child.g+compute_c(child, pred_node,edges))
                    pred_node.rhs=min(tt)
            update_vertex(path, pred_node)
            
            k_old=top_key(path.queue)
            path.start.key=calculateKey(path.start, path.start, path.k_m)
    
    return 'Path found!'
            
            
def get_path(path,graph, G,edges,edges_old=None,change=False,change_list=[]):

    route=[]
    group_numbers=[]
    nodes_indices=[]
    turn_indices=[]
    tmp=(path.start.x,path.start.y,path.start.z)
    route.append(tmp)
    group=path.start.group
    nodes_indices.append(path.start.key_index)
    while path.start.key_index!=path.goal.key_index :

        current_node=path.start
        minim=float('inf')
        for ch in path.start.children:
            n=graph[ch]
           
            if compute_c(path.start, n,edges)+n.g<minim:
                minim=compute_c(path.start, n,edges)+n.g
                current_node=n
                
        if change: #Scan for changes

            path.k_m=path.k_m+heuristic(current_node, path.start)
            #path.start=current_node ###############not sure for that
            
            for c in change_list:
                c_old=compute_c(c[0], c[1],edges_old)

                #update cost and obstacles here
                if c_old>compute_c(c[0], c[1],edges): #if cost is decreased
                    if(c[0]!=path.goal):

                        c[0].rhs=min(c[0].rhs,compute_c(c[0], c[1], edges)+c[1].g)
                        
                elif c[0].rhs== c_old+c[1].g: #if cost is increased
                    if c[0]!=path.goal:

 
                        tt=[]
                        for ch in c[0].children:
                            child=graph[ch]
                            tt.append(child.g+compute_c( c[0],child, edges))
                        c[0].rhs=min(tt)
                update_vertex(path, c[0])
                path.start.g=float('inf')## not sure for that
                path.start.rhs=float('inf')## not sure for that
                path_found=compute_shortest_path(path,graph,G,edges)
                
                print(path_found)
                if not path_found:
                    break
                change=False    
                
        if current_node.key_index!=path.start.key_index:
            #find the intermediate points
            pp=1
            linestring=edges[path.start.key_index][current_node.key_index].geometry #if the start index should go first need to get checked
            coords = list(linestring.coords)
            for c in range(len(coords)-1):
                if not c==0:
                    tmp=(coords[c][0],coords[c][1],current_node.z) #the intermediate point
                    route.append(tmp) 
                    group_numbers.append(current_node.group)
                    nodes_indices.append(-1)
                

            nodes_indices.append(current_node.key_index)
            tmp=(current_node.x,current_node.y,current_node.z) #the next node
            group_numbers.append(current_node.group)
            route.append(tmp) 
            

        path.start=current_node
        
        
    turns=[0]
    for i in range(len(group_numbers)-1):
        if group_numbers[i]==group_numbers[i+1]:
            turns.append(0)
        else:
            turns.append(1)
            turn_indices.append(nodes_indices[i+1])
    turns.append(0)
    turn_indices.append(-1)

    return route,turns,nodes_indices,turn_indices
        
class PathPlanning:
    
    def __init__(self,flow_control_graph,lon_start,lat_start,lon_dest,lat_dest):
        self.start_index=None
        self.goal_index=None
        self.G = None#G
        self.edge_gdf=None#copy.deepcopy(edges)
        self.path=None
        self.os_keys_dict_succ={}
        self.os_keys_dict_pred={}
        self.os_keys_dict={}
        self.route=[]
        self.turns=[]
        self.flow_control_graph=flow_control_graph


        exp_const=0.05##0.005 ## we need to think about the value of that constant
        box=bbox(min(lat_start,lat_dest)-exp_const,min(lon_start,lon_dest)-exp_const,max(lat_start,lat_dest)+exp_const,max(lon_start,lon_dest)+exp_const) 
        
        self.G,edges=self.flow_control_graph.extract_subgraph(box)
        self.edge_gdf=copy.deepcopy(edges)

        d_start,self.start_index=self.flow_control_graph.get_nearest_node(lon_start,lat_start)
        d_dest,self.goal_index=self.flow_control_graph.get_nearest_node(lon_dest,lat_dest)
        
        
        #Create the graph
        self.graph=[]
        omsnx_keys_list=list(self.G.keys())
        
        new_nodes_counter=0
        for i in range(len(omsnx_keys_list)):
           key=omsnx_keys_list[i]
           x=self.G[key].x
           y=self.G[key].y 

           parents=self.G[key].parents
           children=self.G[key].children
           my_group={}
        
           ii=0
           tmp=[]#list if the groups that the node has been added
           for p in parents:
               if not ii:
                   if p in list(self.edge_gdf.keys()) and key in self.edge_gdf[p]: 

                       group=self.edge_gdf[p][key].stroke_group
                       z=self.edge_gdf[p][key].layer_alt
                       node=Node(key,x,y,z,i+new_nodes_counter,group)
                       my_group.update({i+new_nodes_counter:group})
                       self.graph.append(node)
                       tmp.append(group)
                       ii=ii+1
                       if key in self.os_keys_dict_pred.keys():
                           self.os_keys_dict_pred[key][p]=i+new_nodes_counter
                       else:
                           dict={}
                           dict[p]=i+new_nodes_counter
                           self.os_keys_dict_pred[key]=dict
               else: 
                if p in list(self.edge_gdf.keys()) and key in self.edge_gdf[p]: 

                        new_nodes_counter=new_nodes_counter+1
                        group=self.edge_gdf[p][key].stroke_group
                        z=self.edge_gdf[p][key].layer_alt
                        node=Node(key,x,y,z,i+new_nodes_counter,group)
                        my_group.update({i+new_nodes_counter:group})
                        self.graph.append(node)
                        tmp.append(group)
                        ii=ii+1
                        if key in self.os_keys_dict_pred.keys():
                           self.os_keys_dict_pred[key][p]=i+new_nodes_counter
                        else:
                           dict={}
                           dict[p]=i+new_nodes_counter
                           self.os_keys_dict_pred[key]=dict
                           
           for ch in children:
                group=self.edge_gdf[key][ch].stroke_group
                if not group in tmp:
                    if not ii:
                        z=self.edge_gdf[key][ch].layer_alt
                        node=Node(key,x,y,z,i+new_nodes_counter,group)
                        my_group.update({i+new_nodes_counter:group})
                        
                        self.graph.append(node)
                        tmp.append(group)
                        ii=ii+1
                        if key in self.os_keys_dict_succ.keys():
                           self.os_keys_dict_succ[key][ch]=i+new_nodes_counter
                        else:
                           dict={}
                           dict[ch]=i+new_nodes_counter
                           self.os_keys_dict_succ[key]=dict
                        
                    else:
                        new_nodes_counter=new_nodes_counter+1
                        z=self.edge_gdf[key][ch].layer_alt
                        node=Node(key,x,y,z,i+new_nodes_counter,group)
                        my_group.update({i+new_nodes_counter:group})
                        self.graph.append(node)
                        tmp.append(group)
                        ii=ii+1
                        if key in self.os_keys_dict_succ.keys():
                           self.os_keys_dict_succ[key][ch]=i+new_nodes_counter
                        else:
                           dict={}
                           dict[ch]=i+new_nodes_counter
                           self.os_keys_dict_succ[key]=dict
                        
           if ii==0:
                z=10 ########Nees to find what altitude that shoudl be 
                node=Node(key,x,y,z,i+new_nodes_counter,-1)
                self.graph.append(node)

                self.os_keys_dict[key]=i+new_nodes_counter

                        
           if len(my_group)>1:
               for index in my_group:
                   for index_ in my_group:
                        if my_group[index]!=my_group[index_] and index!=index_:
                            self.graph[index].children.append(index_)
                            self.graph[index].parents.append(index_)
                      
        #add the children and parents to each node                
        for i in self.graph:
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
        
        
                    
    def plan(self):
                
        start_id=-1
        goal_id=-1

        for i in self.graph:
            if i.key_index==self.start_index:
                start_id=i.index
            if i.key_index==self.goal_index:
                goal_id=i.index
            if start_id!=-1 and goal_id!=-1:
                break
        print(start_id)
        start_node=self.graph[start_id] 
        x_start=start_node.x
        y_start=start_node.y
        
        goal_node=self.graph[goal_id] 
        
        x_goal=goal_node.x
        y_goal=goal_node.y
         
        
        
        self.path=Path(start_node,goal_node)
        
        initialise(self.path)
        
        path_found=compute_shortest_path(self.path,self.graph, self.G,self.edge_gdf)
        print(path_found)
        
        route=[]
        turns=[]
        edges_list=[]
        next_turn_point=[]
        indices_nodes=[]
        turn_indices=[]
        if path_found:
            route,turns,indices_nodes,turn_indices=get_path(self.path,self.graph, self.G,self.edge_gdf)
            
            edges_list.append(-1)#that is the edge to go to the first point, so no edge exists
            os_id1=indices_nodes[0]
            cnt=0
            for i in range(1,len(indices_nodes)):
                if indices_nodes[i]==-1 or indices_nodes[i]==os_id1:
                    cnt=cnt+1
                else:
                    for j in range(cnt):
                        edges_list.append((os_id1,indices_nodes[i]))
                    edges_list.append((os_id1,indices_nodes[i]))
                    cnt=0
                    os_id1=indices_nodes[i]
            cnt=0
            for i in turns:
                if i:
                    next_turn_point.append(turn_indices[cnt])
                    cnt=cnt+1
                else:
                    next_turn_point.append(turn_indices[cnt])
                    
        self.route=route
        self.turns=turns 
        
        return route,turns,edges_list,next_turn_point
    
    def replan(self,edges_g,current_point_index,index_change_list):
        start_id=-1
        goal_id=-1

  
        for i in self.graph:
            if i.key_index==current_point_index:
                start_id=i.index
                break

        start_node=self.graph[start_id] 
        x_start=start_node.x
        y_start=start_node.y
        
        self.path.start=start_node
        
        edges_g=copy.deepcopy(self.edge_gdf)
        change_list=[]
        for t in index_change_list:
            if t[0] in self.os_keys_dict_succ.keys() and t[1] in self.os_keys_dict_succ[t[0]].keys():
                tmp=[]
                ind=self.os_keys_dict_succ[t[0]][t[1]]
                print(ind)
                tmp.append(self.graph[ind])
                ind=self.os_keys_dict_pred[t[1]][t[0]]
                print(ind)
                tmp.append(self.graph[ind])
                change_list.append(tmp)
                edges_g[t[0]][t[1]].speed=t[2]


        if change_list!=[]: ##get new path only if tehre are changes in the aircraft's subgraph
            #print(self.edge_gdf[index_change_list[0][0]][index_change_list[0][1]].max_speed)
            route=[]
            turns=[]
            route,turns=get_path(self.path,self.graph, self.G,edges_g,self.edge_gdf,True,change_list)
            self.edge_gdf=copy.deepcopy(edges_g)
            
            self.route=route
            self.turns=turns
            
        return self.route,self.turns
        
        