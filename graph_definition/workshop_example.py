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


plan = PathPlanning(graph,gdf,16.3314843,48.2208402 ,16.3424207 ,48.2172288  )
route,turns,edges,next_turn,groups=plan.plan()
x_list=[]
y_list=[]
    
for r in route:
    x_list.append(r[0])
    y_list.append(r[1])
    
ax.scatter(x_list,y_list, color='b')


id1=33144491
id2=283324413
value=1.0
change_list=[[id1,id2,value]]


next_index=33144648
prev_index=33144601

lat= 48.2214613
lon=16.3317914
route,turns,edges,next_turn,groups=plan.replan(change_list,graph,prev_index,next_index,lat,lon)
x_list=[]
y_list=[]
    
for r in route:
    x_list.append(r[0])
    y_list.append(r[1])
    
ax.scatter(x_list,y_list, color='g')

ax.scatter(G._node[id1]['x'],G._node[id1]['y'], color='r')
ax.scatter(G._node[id2]['x'],G._node[id2]['y'], color='r')