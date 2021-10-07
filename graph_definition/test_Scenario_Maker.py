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

# generate some traffic
path_1 = [251523470, 33144695, 30, 0]
path_2 = [25280685,  33144821, 30, 0]

paths = [path_1, path_2]

generated_traffic = bst.Paths2Traf(G, paths)
cruise_speed_constraint = False

# Step 1: Loop through traffic, find path, add to dictionary
scenario_dict = dict()
flight_plans_dict={}
for flight in generated_traffic:
    # First get the route and turns
    origin = flight[2]
    destination = flight[3]

    plan = PathPlanning(graph,gdf, origin[1], origin[0], destination[1], destination[0])
    route=[]
    turns=[]
    route,turns,edges,next_turn,groups=plan.plan()

    flight_plans_dict[flight[0]]=plan
    if route!=[]:
        route = np.array(route)
        # Create dictionary
        scenario_dict[flight[0]] = dict()
        # Add start time
        scenario_dict[flight[0]]['start_time'] = flight[1]
        #Add lats
        scenario_dict[flight[0]]['lats'] = route[:,1]
        #Add lons
        scenario_dict[flight[0]]['lons'] = route[:,0]
        #Add turnbool
        scenario_dict[flight[0]]['turnbool'] = turns
        #Add alts
        scenario_dict[flight[0]]['alts'] = []
        #Add active edges
        scenario_dict[flight[0]]['edges'] = edges
        #Add stroke group
        scenario_dict[flight[0]]['stroke_group'] = groups
        #Add next turn
        scenario_dict[flight[0]]['next_turn'] = next_turn
    
    

    
print('All paths created!')
    
# Step 4: Create scenario file from dictionary
bst.Dict2Scn(r'cr_test1.scn', 
              scenario_dict, 'fake_path', cruise_speed_constraint, start_speed=30)

print('Scenario file created!')