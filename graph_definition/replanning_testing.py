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


plan = PathPlanning(graph,gdf,  16.3416951,48.2173994,16.3409908, 48.2274167 )

route,turns,edges,next_turn,groups=plan.plan()
x_list=[]
y_list=[]
    
for r in route:
    x_list.append(r[0])
    y_list.append(r[1])
    
ax.scatter(x_list,y_list, color='b')
ax.scatter( 16.3416951,48.2173994, color='g')

id1=29048466
id2=283324405
value=1.0

graph.edges_graph[id1][id2].speed=value




change_list=[[id1,id2,value]]


next_index=3312560802
ax.scatter(G._node[next_index]['x'],G._node[next_index]['y'], color='r')
lat= 48.2173994 
lon=16.3416951

route,turns,edges,next_turn,groups=plan.replan(change_list,graph,next_index,lat,lon)
x_list=[]
y_list=[]
    
for r in route:
    x_list.append(r[0])
    y_list.append(r[1])
    
ax.scatter(x_list,y_list, color='g')

ax.scatter(G._node[id1]['x'],G._node[id1]['y'], color='r')
ax.scatter(G._node[id2]['x'],G._node[id2]['y'], color='r')
