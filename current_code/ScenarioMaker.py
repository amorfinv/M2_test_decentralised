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
#G = ox.io.load_graphml('processed_graph.graphml')
edges = ox.graph_to_gdfs(G)[1]
gdf=ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
print('Graph loaded!')

# read line by line of file
flight_intention_list  = []
flight_intention_folder = 'flight_intentions/'
intention_file_name = 'Flight_intention_ultra_40_0.csv'
print(intention_file_name[0:~3])
with open(flight_intention_folder + intention_file_name) as file:
    for line in file:
        line = line.strip()
        line = line.split(',')
        flight_intention_list.append(line)

# Step 2: Generate traffic from the flight intention file
generated_traffic, loitering_edges_dict = bst.Intention2Traf(flight_intention_list, edges.copy())
print('Traffic generated!')

#Load the open airspace grid
input_file=open("renamed_open_airspace_grid.dill", 'rb')
#input_file=open("open_airspace_grid_updated.dill", 'rb')##for 3d path planning
grid=dill.load(input_file)


##Initialise the flow control entity
graph=street_graph(G,edges,grid) 

# create a loitering dill
scenario_loitering_dill_folder = 'scenario_loitering_dills/'
scenario_loitering_dill_name = intention_file_name.replace('.csv','.dill')
output_file=open(scenario_loitering_dill_folder + scenario_loitering_dill_name, 'wb')
dill.dump(loitering_edges_dict,output_file)
output_file.close()
print('Created loitering dill')

#fig, ax = ox.plot_graph(G,node_color="w",show=False,close=False)
s=0

plan=None
# Step 3: Loop through traffic, find path, add to dictionary
scenario_dict = dict()
sizes=[]
lines = []

for cnt, flight in enumerate(generated_traffic):
    print(flight[0])

    if cnt>=100:#0 :
        break #stop at 20 aircrafts or change that

    # First get the route and turns
    origin = flight[3]
    destination = flight[4]

    priority = flight[8]
    aircraft_type = 1 if flight[1] == 'MP20' else 2
    

    plan = PathPlanning(aircraft_type,grid,graph,gdf, origin[0], origin[1], destination[0], destination[1])
    route,turns,edges,next_turn,groups,in_constrained,turn_speed=plan.plan()

    if route==[]: ##TODO convert that to a while and 
        #No path was found
        plan = PathPlanning(aircraft_type,grid,graph,gdf, origin[0], origin[1], destination[0], destination[1],0.03)
        route,turns,edges,next_turn,groups,in_constrained,turn_speed=plan.plan()

    # Step 4: Add to dictionary
    file_loc_dill = flight[5]
    # TODO: delet here after we have all the DILLS!!!
    
    del plan.flow_graph
    output_file=open(f"path_plan_dills/{file_loc_dill}.dill", 'wb')
    dill.dump(plan,output_file)
    output_file.close()
    #############################################
    
    # Get the flight intention information
    drone_id = flight[0]
    aircraft_type = flight[1]
    start_time = flight[2]
    origin_lat = flight[3][1]
    origin_lon = flight[3][0]
    dest_lat = flight[4][1]
    dest_lon = flight[4][0]
    file_loc = flight[5]
    priority = flight[7]
    geoduration = flight[8]
    geocoords = flight[9] 

    # constants for scenario
    start_speed = 0.0
    qdr = 0.0
    alt = 0.0

    # Convert start_time to Bluesky format
    start_time = round(start_time)
    m, s = divmod(start_time, 60)
    h, m = divmod(m, 60)
    start_time_txt = f'{h:02d}:{m:02d}:{s:02d}>'

    # QUEUE COMMAND
    if geocoords:
        queue_text = f'QUEUEM2 {drone_id},{aircraft_type},{file_loc},{origin_lat},{origin_lon},{dest_lat},{dest_lon},{qdr},{alt},{start_speed},{priority},{geoduration},{geocoords}\n'
    else:
        queue_text = f'QUEUEM2 {drone_id},{aircraft_type},{file_loc},{origin_lat},{origin_lon},{dest_lat},{dest_lon},{qdr},{alt},{start_speed},{priority},{geoduration},\n'
    
    lines.append(start_time_txt + queue_text)
    
    
# write stuff to file
scenario_folder = 'scenarios/'
scenario_file_name = intention_file_name.replace('csv','scn')

# Step 4: Create scenario file from dictionary
with open(scenario_folder+scenario_file_name, 'w+') as f:
    f.write('00:00:00>HOLD\n00:00:00>PAN 48.204011819028494 16.363471515762452\n00:00:00>ZOOM 10\n')
    f.write('00:00:00>ASAS ON\n00:00:00>RESO SPEEDBASEDV2\n00:00:00>CDMETHOD M2STATEBASED\n')
    f.write('00:00:00>STREETSENABLE\n')
    f.write(f'00:00:00>loadloiteringdill {scenario_loitering_dill_name[0:~3]}\n')
    f.write('00:00:00>CASMACHTHR 0\n')
    f.write(''.join(lines))
    
print('Scenario file created!')
