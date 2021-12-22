# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 17:00:22 2021

@author: nipat
"""
import osmnx as ox
import numpy as np
import BlueskySCNTools
from plugins.streets.flow_control import street_graph,bbox
from plugins.streets.agent_path_planning import PathPlanning,Path
from plugins.streets.open_airspace_grid import Cell, open_airspace
import os
import dill
import json
import sys
from pympler import asizeof


# Step 1: Import the graph we will be using
dir_path = os.path.dirname(os.path.realpath(__file__))
graph_path = dir_path.replace('current_code', 
          'current_code/whole_vienna/gis/finalized_graph.graphml')
G = ox.io.load_graphml(graph_path)
#G = ox.io.load_graphml('processed_graph.graphml')
edges = ox.graph_to_gdfs(G)[1]
gdf=ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
print('Graph loaded!')



#Load the open airspace grid
input_file=open("renamed_open_airspace_grid.dill", 'rb')
#input_file=open("open_airspace_grid_updated.dill", 'rb')##for 3d path planning
grid=dill.load(input_file)


##Initialise the flow control entity
graph=street_graph(G,edges,grid) 

fig, ax = ox.plot_graph(G,node_color="w",show=False,close=False)

plan = PathPlanning(2,grid,graph,gdf, 16.3483711835,48.2054864165,16.3608620862,48.2356842079)
route,turns,edges,next_turn,groups,in_constrained,turn_speed=plan.plan()

x_list=[]
y_list=[]
for r in route:
    x_list.append(r[0])
    y_list.append(r[1])

ax.scatter(x_list,y_list,c="b")




 
