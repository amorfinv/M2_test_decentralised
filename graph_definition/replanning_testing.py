 # -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 14:37:42 2021

@author: nipat
"""

import osmnx as ox
import numpy as np
import BlueskySCNTools
from plugins.streets.flow_control import street_graph,bbox
from plugins.streets.agent_path_planning import PathPlanning,Path
import os
import dill
import time


# Initialize stuff
bst = BlueskySCNTools.BlueskySCNTools()

# Step 1: Import the graph we will be using
dir_path = os.path.dirname(os.path.realpath(__file__))
graph_path = dir_path.replace('graph_definition', 
          'graph_definition/gis/data/street_graph/processed_graph.graphml')
G = ox.io.load_graphml(graph_path)
#G = ox.io.load_graphml('processed_graph.graphml')
edges = ox.graph_to_gdfs(G)[1]
gdf=ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
print('Graph loaded!')

##Initialise the flow control entity
graph=street_graph(G,edges) 

fig, ax = ox.plot_graph(G,node_color="w",show=False,close=False)

#plan = PathPlanning(graph,gdf,16.328936,48.221698 ,16.3436724,48.226886  )
plan = PathPlanning(graph,gdf,16.3314843 ,48.221698 ,16.3335919,48.2293961)
#plan = PathPlanning(graph,gdf,16.343225, 48.2182985 ,16.3335919 ,48.2293961 )#d1
#plan = PathPlanning(graph,gdf,16.3498014,48.2250679 ,16.3378102  ,48.2187331 )#d2
#plan = PathPlanning(graph,gdf,16.3314843 ,48.2208402 ,16.3392305  ,48.2176641)#d3

seconds = time.time()
route,turns,edges,next_turn,groups=plan.plan()
s = time.time()
print(s-seconds)
x_list=[]
y_list=[]
    
for r in route:
    x_list.append(r[0])
    y_list.append(r[1])
    
ax.scatter(x_list,y_list, color='b')
ax.scatter(x_list[0],y_list[0], color='g')
ax.scatter(x_list[len(x_list)-1],y_list[len(x_list)-1], color='r')

id1=33345291 
id2=251523470
value=5.0

# =============================================================================
# 
# ax.scatter(G._node[id1]['x'],G._node[id1]['y'], color='r')
# ax.scatter(G._node[id2]['x'],G._node[id2]['y'], color='r')
# =============================================================================

graph.edges_graph[id1][id2].speed=value

change_list=[[id1,id2,value]]


next_index=33144621
prev_index=33144616
#ax.scatter(G._node[next_index]['x'],G._node[next_index]['y'], color='r')
lat= 48.2219805
lon=16.3281112 


# =============================================================================
# ##d1
# next_index=33143898
# prev_index=3312560802
# #ax.scatter(G._node[next_index]['x'],G._node[next_index]['y'], color='r')
# lat= 48.2185219
# lon=16.3419587
# 
# =============================================================================
# =============================================================================
# ##d2
# next_index=213287623 
# prev_index=378727
# #ax.scatter(G._node[next_index]['x'],G._node[next_index]['y'], color='r')
# lat= 48.2249578
# lon=16.3497759
# =============================================================================

# =============================================================================
# ##d3
# next_index=33144637
# prev_index=33144640
# #ax.scatter(G._node[next_index]['x'],G._node[next_index]['y'], color='r')
# lat= 48.2208263
# lon=16.3339504
# =============================================================================

# =============================================================================
# print("replan")
# 
# seconds = time.time()
# 
# route,turns,edges,next_turn,groups=plan.replan(change_list,graph,prev_index,next_index,lat,lon)
# s = time.time()
# print(s-seconds)
# 
# x_list=[]
# y_list=[]
#     
# for r in route:
#     x_list.append(r[0])
#     y_list.append(r[1])
#     
# ax.scatter(x_list,y_list, color='g')
# =============================================================================




# =============================================================================
# print("plan3")
# plan1 = PathPlanning(graph,gdf,16.3314843,48.2208402 ,16.3424207 ,48.2172288  )
# 
# plan1 = PathPlanning(graph,gdf,16.3317707,48.2214195 ,16.3424207 ,48.2172288  )
# route,turns,edges,next_turn,groups=plan1.plan()
# x_list=[]
# y_list=[]
#     
# for r in route:
#     x_list.append(r[0])
#     y_list.append(r[1])
#     
# ax.scatter(x_list,y_list, color='b')
# print("plan7")
# plan = PathPlanning(graph,gdf,16.348074, 48.2253325 ,16.3436724  ,48.226886  )
# 
# route,turns,edges,next_turn,groups=plan.plan()
# x_list=[]
# y_list=[]
#     
# for r in route:
#     x_list.append(r[0])
#     y_list.append(r[1])
#     
# ax.scatter(x_list,y_list, color='b')
# ax.scatter( 16.3416951,48.2173994, color='g')
# 
# next_index=33344816
# ax.scatter(G._node[next_index]['x'],G._node[next_index]['y'], color='r')
# next_index=33144648
# ax.scatter(G._node[next_index]['x'],G._node[next_index]['y'], color='r')
# 
# next_index=378727
# #ax.scatter(G._node[next_index]['x'],G._node[next_index]['y'], color='r')
# #the two next nodes
# # =============================================================================
# # next_index=26405238
# # ax.scatter(G._node[next_index]['x'],G._node[next_index]['y'], color='r')
# # next_index=213287623
# # ax.scatter(G._node[next_index]['x'],G._node[next_index]['y'], color='r')
# # =============================================================================
# 
# next_index=1629378075
# ax.scatter(G._node[next_index]['x'],G._node[next_index]['y'], color='g')
# next_index=213287623
# ax.scatter(G._node[next_index]['x'],G._node[next_index]['y'], color='g')
# 
# 
# next_index=394751
# #ax.scatter(G._node[next_index]['x'],G._node[next_index]['y'], color='r')
# 
# id1=213287623
# id2=1629378075
# value=0.0
# 
# 
# ax.scatter(G._node[id1]['x'],G._node[id1]['y'], color='r')
# ax.scatter(G._node[id2]['x'],G._node[id2]['y'], color='r')
# 
# graph.edges_graph[id1][id2].speed=value
# 
# change_list=[[id1,id2,value]]
# 
# 
# next_index=394751
# prev_index=1119870220
# #ax.scatter(G._node[next_index]['x'],G._node[next_index]['y'], color='r')
# lat= 48.22523732955928 
# lon=16.348427007274488
# 
# print("replan7")
# route,turns,edges,next_turn,groups=plan.replan(change_list,graph,prev_index,next_index,lat,lon)
# 
# x_list=[]
# y_list=[]
#     
# for r in route:
#     x_list.append(r[0])
#     y_list.append(r[1])
#     
# ax.scatter(x_list,y_list, color='g')
# 
# next_index1=33144648 
# prev_index1=33144601
# 
# lat1=48.22110395207955 
# lon1=16.331614695485165
# print("replan3")
# route,turns,edges,next_turn,groups=plan1.replan(change_list,graph,prev_index1,next_index1,lat1,lon1)
# 
# x_list=[]
# y_list=[]
#     
# for r in route:
#     x_list.append(r[0])
#     y_list.append(r[1])
#     
# ax.scatter(x_list,y_list, color='g')
# =============================================================================

# =============================================================================
# plan = PathPlanning(graph,gdf,  16.3416951,48.2173994,16.3409908, 48.2274167 )
# 
# route,turns,edges,next_turn,groups=plan.plan()
# x_list=[]
# y_list=[]
#     
# for r in route:
#     x_list.append(r[0])
#     y_list.append(r[1])
#     
# ax.scatter(x_list,y_list, color='b')
# ax.scatter( 16.3416951,48.2173994, color='g')
# 
# id1=29048466
# id2=283324405
# value=1.0
# 
# graph.edges_graph[id1][id2].speed=value
# 
# 
# 
# 
# change_list=[[id1,id2,value]]
# 
# 
# next_index=3312560802
# ax.scatter(G._node[next_index]['x'],G._node[next_index]['y'], color='r')
# lat= 48.2173994 
# lon=16.3416951
# 
# route,turns,edges,next_turn,groups=plan.replan(change_list,graph,next_index,lat,lon)
# x_list=[]
# y_list=[]
#     
# for r in route:
#     x_list.append(r[0])
#     y_list.append(r[1])
#     
# ax.scatter(x_list,y_list, color='g')
# 
# ax.scatter(G._node[id1]['x'],G._node[id1]['y'], color='r')
# ax.scatter(G._node[id2]['x'],G._node[id2]['y'], color='r')
# =============================================================================
