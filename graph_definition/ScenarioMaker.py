# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:47:30 2021

@author: andub
"""
import osmnx as ox
import numpy as np
import BlueskySCNTools
from flow_control import street_graph,bbox
from agent_path_planning import PathPlanning
import os

# Initialize stuff
bst = BlueskySCNTools.BlueskySCNTools()

# Step 1: Import the graph we will be using
dir_path = os.path.dirname(os.path.realpath(__file__))
#graph_path = dir_path.replace('graph_definition', 
#          'graph_definition/gis/data/street_graph/processed_graph.graphml')
#G = ox.io.load_graphml(graph_path)
G = ox.io.load_graphml('processed_graph.graphml')
edges = ox.graph_to_gdfs(G)[1]
print('Graph loaded!')

##Initialise the flow control entity
graph=street_graph(G,edges) 

# Step 2: Generate traffic from it
concurrent_ac = 10
aircraft_vel = 12 # [m/s]
max_time = 600 # [s]
dt = 10
min_dist = 1000 # [m]

orig_nodes = [30696015, 3155094143, 33345321,  25280685, 1119870220, 33302019,
              33144416, 378696, 33143911, 264055537, 33144706, 33144712, 
              33144719, 92739749]

dest_nodes = [291088171,  60957703, 30696019, 392251, 33301346, 26405238, 
              3963787755, 33345333, 378699, 33144821, 264061926, 33144695,
              33174086, 33345331]

generated_traffic = bst.Graph2Traf(G, concurrent_ac, aircraft_vel, max_time, 
                                   dt, min_dist, orig_nodes, dest_nodes)
print('Traffic generated!')

# Step 3: Loop through traffic, find path, add to dictionary
scenario_dict = dict()
if 1:
    flight = generated_traffic[0]
#for flight in generated_traffic:
    # First get the route and turns
    origin = flight[2]
    destination = flight[3]
    plan = PathPlanning(graph, origin[1], origin[0], destination[1], destination[0])
    route=[]
    turns=[]
    route,turns,edges,next_turn=plan.plan()
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
        scenario_dict[flight[0]]['alts'] = route[:,2]
    
print('All paths created!')
    
# Step 4: Create scenario file from dictionary
bst.Dict2Scn(r'Test_Scenario.scn', 
             scenario_dict)

print('Scenario file created!')
    