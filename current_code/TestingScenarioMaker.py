# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:47:30 2021
@author: andub
"""
from datetime import time
import osmnx as ox
import numpy as np
import BlueskySCNTools
from plugins.streets.flow_control import street_graph,bbox
from plugins.streets.agent_path_planning import PathPlanning,Path
from plugins.streets.open_airspace_grid import Cell, open_airspace
import os
import dill
import json

scenario_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),'custom_scenarios')

## read in origin and destinations from origin_destination.json
with open('origin_destination.json') as f:
    origin_destination_dict = json.load(f)

origins_dict = origin_destination_dict['origins']
destinations_dict = origin_destination_dict['destinations']
center_dict = origin_destination_dict['center']

###############################################################################
# # create an overtake scenario

test_scenario_name = 'overtake_scenario.scn'
path_plan_filename = 'overtake_scenario'
loitering_name = 'overtake_loitering'

# put two aircraft at 30 ft, one is ac1 is at 20 kts and ac2 is at 30 kts
ac_1 = {'ac_type': 'MP20','origin': center_dict['475'], 'destination': center_dict['318'], 'start_time': 0, 
        'priority': 1, 'start_speed': 20, 'altitude': 30,'geoduration': 0, 'geocoords': None}

ac_2 = {'ac_type': 'MP30','origin': center_dict['475'], 'destination': center_dict['318'], 'start_time': 10, 
        'priority': 1, 'start_speed': 30, 'altitude': 30, 'geoduration': 0, 'geocoords': None}

flight_intention_list = [ac_1, ac_2]
# ################################################################################

###############################################################################
# create an overtake scenario with an aircraft above

# test_scenario_name = 'overtake_scenario2.scn'
# path_plan_filename = 'overtake_scenario2'
# loitering_name = 'overtake_loitering2'

# # put two aircraft at 30 ft, one is ac1 is at 20 kts and ac2 is at 30 kts
# ac_1 = {'ac_type': 'MP20', 'origin': center_dict['475'], 'destination': center_dict['318'], 'start_time': 0, 
#         'priority': 1, 'start_speed': 20, 'altitude': 30,'geoduration': 0, 'geocoords': None}

# ac_2 = {'ac_type': 'MP30', 'origin': center_dict['475'], 'destination': center_dict['318'], 'start_time': 10, 
#         'priority': 1, 'start_speed': 30, 'altitude': 30, 'geoduration': 0, 'geocoords': None}

# # put third aircraft at 60 ft
# ac_3 = {'ac_type': 'MP30', 'origin': center_dict['475'], 'destination': center_dict['318'], 'start_time': 10,
#         'priority': 1, 'start_speed': 30, 'altitude': 60, 'geoduration': 0, 'geocoords': None}

# flight_intention_list = [ac_1, ac_2, ac_3]
###############################################################################

# Initialize stuff
bst = BlueskySCNTools.BlueskySCNTools()

# Step 1: Import the graph we will be using
dir_path = os.path.dirname(os.path.realpath(__file__))
graph_path = dir_path.replace('current_code', 
          'current_code/whole_vienna/gis/layer_heights.graphml')
G = ox.io.load_graphml(graph_path)
#G = ox.io.load_graphml('processed_graph.graphml')
edges = ox.graph_to_gdfs(G)[1]
gdf=ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
print('Graph loaded!')

#Load the open airspace grid
use_open_grid=True#False
if use_open_grid:
    input_file=open("open_airspace_grid.dill", 'rb')
    grid=dill.load(input_file)
else:
    grid=[]

##Initialise the flow control entity
graph=street_graph(G,edges) 

# Step 2: Generate traffic from the flight intention file
generated_traffic, loitering_edges_dict = bst.TestIntention2Traf(flight_intention_list, edges.copy())
print('Traffic generated!')

# create a loitering dill
fpath_loitering = os.path.join(scenario_folder, loitering_name)
output_file=open(f"{fpath_loitering}.dill", 'wb')
dill.dump(loitering_edges_dict,output_file)
output_file.close()
print('Created loitering dill')


aircraft_type=1
priority=1
# Step 3: Loop through traffic, find path, add to dictionary
scenario_dict = dict()
flight_plans_dict={}
for flight in generated_traffic:
    # First get the route and turns
    origin = flight[3]
    destination = flight[4]
    
    if flight[0] in loitering_edges_dict.keys():
        plan = PathPlanning(aircraft_type,priority,grid,graph,gdf, origin[0], origin[1], destination[0], destination[1],True,loitering_edges_dict[flight[0]])
    else:
        plan = PathPlanning(aircraft_type,priority,grid,graph,gdf, origin[0], origin[1], destination[0], destination[1])
    route,turns,edges,next_turn,groups,in_constrained,turn_speed=plan.plan()

    flight_plans_dict[flight[0]]=plan
    if route!=[]:
        route = np.array(route)
        # Create dictionary
        scenario_dict[flight[0]] = dict()
        # add aircraft type
        scenario_dict[flight[0]]['aircraft_type'] = flight[1]
        # Add start time
        scenario_dict[flight[0]]['start_time'] = flight[2]
        #Add lats
        scenario_dict[flight[0]]['lats'] = route[:,1]
        #Add lons
        scenario_dict[flight[0]]['lons'] = route[:,0]
        # get start speed
        scenario_dict[flight[0]]['start_speed'] = flight[5]
        #Add turnbool
        scenario_dict[flight[0]]['turnbool'] = turns
        #Add alts
        scenario_dict[flight[0]]['alts'] = flight[6]
        #Add active edges
        scenario_dict[flight[0]]['edges'] = edges
        #Add stroke group
        scenario_dict[flight[0]]['stroke_group'] = groups
        #Add next turn
        scenario_dict[flight[0]]['next_turn'] = next_turn
        #Add constarined airspace indicator
        scenario_dict[flight[0]]['airspace_type'] = in_constrained
        #add priority
        scenario_dict[flight[0]]['priority'] = flight[7]
        # add geoduration
        scenario_dict[flight[0]]['geoduration'] = flight[8]
        # add geocoords
        scenario_dict[flight[0]]['geocoords'] = flight[9]
    
    


print('All paths created!')
    
# Step 4: Create scenario file from dictionary
fpath_scn = os.path.join(scenario_folder, test_scenario_name)
bst.Dict2Scn(f'{fpath_scn}', 
              scenario_dict, path_plan_filename)

print('Scenario file created!')

list2dill=[]
list2dill.append(flight_plans_dict)
list2dill.append(graph)

##Dill the flight_plans_dict
fpath_path = os.path.join(scenario_folder, path_plan_filename)
output_file=open(f"{fpath_path}.dill", 'wb')
dill.dump(list2dill,output_file)
output_file.close()

#output_file=open("G-multigraph.dill", 'wb')
#dill.dump(G,output_file)
#output_file.close()

print("Flight plans and search graphs saved!")
