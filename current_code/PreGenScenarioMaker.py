# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:47:30 2021
@author: andub
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

# Initialize stuff
bst = BlueskySCNTools.BlueskySCNTools()

# Step 1: Import the graph we will be using
dir_path = os.path.dirname(os.path.realpath(__file__))
graph_path = dir_path.replace('current_code', 
          'current_code/whole_vienna/gis/renamed_graph.graphml')
G = ox.io.load_graphml(graph_path)
edges = ox.graph_to_gdfs(G)[1]
gdf=ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
print('Graph loaded!')

#Load the open airspace grid
input_file=open("renamed_open_airspace_grid.dill", 'rb')
grid=dill.load(input_file)

##Initialise the flow control entity
graph=street_graph(G,edges,grid) 

fig, ax = ox.plot_graph(G,node_color="w",show=False,close=False)
s=0

plan=None
# Step 3: Loop through traffic, find path, add to dictionary
scenario_dict = dict()
cnt=0
flight_plans_dict={}
sizes=[]

aircraft_types = ['MP20', 'MP30']

# create dills for two aircraft types
for aircraft_type in aircraft_types:

    # Go through each origin destination pair and save the dill
    # Multiprocess me please
    for file_num, flight in enumerate(bst.pairs_list):

        if file_num>20:#0 :
            break #stop at 20 aircrafts or change that

        # First get the origin, destinations
        origin_lon = flight[0]
        origin_lat = flight[1]

        destination_lon = flight[2]
        destination_lat = flight[3]

        # generate the path planning object
        plan = PathPlanning(aircraft_type, grid, graph,gdf, origin_lon, origin_lat, destination_lon, destination_lat)
        
        route,_,edges,_,_,_,_=plan.plan()
        if route==[]: ##TODO convert that to a while and 
            #No path was found
            plan = PathPlanning(aircraft_type,grid,graph,gdf, origin_lon, origin_lat, destination_lon, destination_lat,0.03)
            
            route,_,edges,_,_,_,_=plan.plan()
            
        if len(route)!=len(edges):
            print("unequal lens",len(route),len(edges))

        print("size",asizeof.asizeof(plan)-asizeof.asizeof(graph))
        s=s+asizeof.asizeof(plan)-asizeof.asizeof(graph)

        # Delete the graph from plan and then save dill   
        del plan.flow_graph

        file_loc_dill = f'{file_num}_{aircraft_type}'
        output_file=open(f"path_plan_dills/{file_loc_dill}.dill", 'wb')
        dill.dump(plan,output_file)
        output_file.close()


print("total size of all dills",s)    
print('All paths created!')
print("Pre-generated plans and search graphs saved!")
