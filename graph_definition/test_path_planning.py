# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:46:47 2021

@author: nipat
"""
import pickle
from flow_control import street_graph,bbox
from agent_path_planning import PathPlanning
import time

start = time.time()
G= pickle.load(open("G-multigraph.pickle", "rb"))#load G
edge=pickle.load(open("edge_gdf.pickle", "rb"))#load edge_geometry

now = time.time()
#print("Loaded pickle")
#print(now - start)
#Create the graph of Vienna
graph=street_graph(G,edge) 

#extract the subgraph for a mission
y_start=48.225 #latitude
x_start=16.34 #longitude
y_dest=48.22
x_dest=16.333

##In the bbox aconstant value should be added to expand it
exp_const=0.005 ## we need to think about the value of that constant
box=bbox(min(y_start,y_dest)-exp_const,min(x_start,x_dest)-exp_const,max(y_start,y_dest)+exp_const,max(x_start,x_dest)+exp_const) 
subgraph,edges=graph.extract_subgraph(box)

now = time.time()
#print("Get subgraph")
#print(now - start)

d_start,start_index=graph.get_nearest_node(x_start,y_start)
d_dest,dest_index=graph.get_nearest_node(x_dest,y_dest)


curr_index1=33143834
curr_index2=33144555
edges[33143834][33144555].max_speed=0

plan1=PathPlanning(graph, x_start,y_start,x_dest,y_dest)
route,turns,edge_list,next_turn= plan1.plan()
#print(route)
print("turns")
print(turns)
print("edge_list")
print(edge_list)
print("next trun")
print(next_turn)

if 0:
    now = time.time()
    print("Planned")
    print(now - start)
    
    ##replan
    y_curr=route[0][3][1]
    x_curr=route[0][3][0]
    d_curr,curr_index=graph.get_nearest_node(x_curr,y_curr)
    #print(curr_index)
    #print("start")
    #curr_index=33143834
    
    y1=route[0][6][1]
    x1=route[0][6][0]
    #d1,curr_index1=graph.get_nearest_node(x1,y1)
    print(curr_index1)
    y2=route[0][7][1]
    x2=route[0][7][0]
    #d2,curr_index2=graph.get_nearest_node(x2,y2)
    print(curr_index2)
    
    ##changes
    #subgraph,edges1=graph.extract_subgraph(box)
    #curr_index1=29048468
    #curr_index2=33144484
    edges[curr_index1][curr_index2].max_speed=10
    
    
    change_list=[[curr_index1,curr_index2]]          
    
    
    ######Blueksy should provide the last vistited and the next to  be visited point for each replanning aircraft
    new_route=plan1.replan(edges,curr_index,change_list)
    
    now = time.time()
    print("Replanned")
    print(now - start)
    
    ######################
    #Visualise
    import osmnx
    
    ind=graph.get_nodes_in_area(box)
    x=[]
    y=[]
    for i in ind:
        x.append(G._node[i]['x'])
        y.append(G._node[i]['y'])
        
    
    fig, ax = osmnx.plot_graph(G,node_color="w",show=False,close=False)
    ax.scatter(x,y, color='r')
    ax.scatter(x_start,y_start, color='g')
    ax.scatter(x_dest,y_dest, color='b')
    
    
    
    x_list=[]
    y_list=[]
    
    for r in route[0]:
        x_list.append(r[0])
        y_list.append(r[1])
    
    ax.scatter(x_list,y_list, color='y')
    ax.scatter(G._node[start_index]['x'],G._node[start_index]['y'], color='g')
    ax.scatter(G._node[dest_index]['x'],G._node[dest_index]['y'], color='b')
    
    ax.scatter(x_curr,y_curr, color='g')
    
    x_new_list=[]
    y_new_list=[]
    
    for r in new_route[0]:
        x_new_list.append(r[0])
        y_new_list.append(r[1])
    
    ax.scatter(x_new_list,y_new_list, color='w')
    ax.scatter(x_curr,y_curr, color='g')