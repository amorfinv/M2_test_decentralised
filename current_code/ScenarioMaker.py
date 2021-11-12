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
input_file=open("open_airspace_grid.dill", 'rb')
grid=dill.load(input_file)



##Initialise the flow control entity
graph=street_graph(G,edges) 


# path planning file TODO: match scenario name
path_plan_filename = 'Path_Planning'

# Step 2: Generate traffic from it
concurrent_ac = 5
aircraft_vel = 12 # [m/s]
max_time = 60 # [s]
dt = 10
min_dist = 1000 # [m]
cruise_speed_constraint = True

with open('origin_destination.json', 'r') as filename:
            origin_dest = json.load(filename)
            
orig_nodes=origin_dest['origins']    
dest_nodes=origin_dest['destinations']   
     
# =============================================================================
# orig_nodes = [30696015, 3155094143, 33345321,  25280685, 1119870220, 33302019,
#               33144416, 378696, 33143911, 264055537, 33144706, 33144712, 
#               33144719, 92739749]
# 
# dest_nodes = [291088171,  60957703, 30696019, 392251, 33301346, 26405238, 
#               3963787755, 33345333, 378699, 33144821, 264061926, 33144695,
#               33174086, 33345331]
# =============================================================================

generated_traffic = bst.Graph2Traf(G, concurrent_ac, aircraft_vel, max_time, 
                                    dt, min_dist, orig_nodes, dest_nodes)
print('Traffic generated!')

# =============================================================================
# lon_start=16.3304374
# lat_start=48.2293708
# lon_dest=16.3507849
# lat_dest=48.224925
# plan = PathPlanning(grid,graph,gdf, lon_start,lat_start,lon_dest,lat_dest)
# route=[]
# turns=[]
# route,turns,edges,next_turn,groups,in_constrained=plan.plan()
# 
# print("planned")
# =============================================================================

aircraft_type=1
priority=1
# Step 3: Loop through traffic, find path, add to dictionary
scenario_dict = dict()
flight_plans_dict={}
for flight in generated_traffic:
    # First get the route and turns
    origin = flight[2]
    destination = flight[3]
    plan = PathPlanning(aircraft_type,priority,grid,graph,gdf, origin[1], origin[0], destination[1], destination[0])
    route,turns,edges,next_turn,groups,in_constrained,turn_speed=plan.plan()
    print(turns)
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
        #scenario_dict[flight[0]]['alts'] = route[:,2]
        scenario_dict[flight[0]]['alts'] = []

        #Add active edges
        scenario_dict[flight[0]]['edges'] = edges
        #Add stroke group
        scenario_dict[flight[0]]['stroke_group'] = groups
        #Add next turn
        scenario_dict[flight[0]]['next_turn'] = next_turn
        #Add constarined airspace indicator
        scenario_dict[flight[0]]['airspace_type'] = in_constrained
    
    


print('All paths created!')
    
# Step 4: Create scenario file from dictionary
bst.Dict2Scn(r'Test_Scenario.scn', 
              scenario_dict, path_plan_filename, cruise_speed_constraint)

print('Scenario file created!')

list2dill=[]
list2dill.append(flight_plans_dict)
list2dill.append(graph)

##Dill the flight_plans_dict
output_file=open(f"{path_plan_filename}.dill", 'wb')
dill.dump(list2dill,output_file)
output_file.close()

#output_file=open("G-multigraph.dill", 'wb')
#dill.dump(G,output_file)
#output_file.close()

print("Flight plans and search graphs saved!")
    
##example of loading the flight plans
#input_file=open("Path_Planning.dill", 'rb')
#p=dill.load(input_file)

