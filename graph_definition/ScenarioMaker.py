# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:47:30 2021

@author: andub
"""
import osmnx as ox
import numpy as np
import BlueskySCNTools
from path_planning import PathPlanning
import os

# Initialize stuff
bst = BlueskySCNTools.BlueskySCNTools()

# Step 1: Import the graph we will be using
dir_path = os.path.dirname(os.path.realpath(__file__))
graph_path = dir_path.replace('graph_definition', 
          'graph_definition/gis/data/street_graph/processed_graph.graphml')
G = ox.io.load_graphml(graph_path)
print('Graph loaded!')

# Step 2: Generate traffic from it
concurrent_ac = 10
aircraft_vel = 12
max_time = 600
dt = 10
min_dist = 1000
generated_traffic = bst.Graph2Traf(G, concurrent_ac, aircraft_vel, max_time, dt, min_dist)
print('Traffic generated!')

# Step 3: Loop through traffic, find path, add to dictionary
scenario_dict = dict()
for flight in generated_traffic:
    # First get the route and turns
    origin = flight[2]
    destination = flight[3]
    plan = PathPlanning(G, origin[1], origin[0], destination[1], destination[0])
    route=[]
    turns=[]
    route,turns=plan.plan()
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
    scenario_dict[flight[0]]['alts'] = route[:,2]
    
print('All paths created!')
    
# Step 4: Create scenario file from dictionary
bst.Dict2Scn(r'C:\Users\andub\Desktop\Bluesky\scenario\Test_Scenario.scn', scenario_dict)

print('Scenario file created!')
    