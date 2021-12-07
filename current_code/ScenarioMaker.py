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
          'current_code/whole_vienna/gis/layer_heights_simplified.graphml')
G = ox.io.load_graphml(graph_path)
#G = ox.io.load_graphml('processed_graph.graphml')
edges = ox.graph_to_gdfs(G)[1]
gdf=ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
print('Graph loaded!')

#Load the open airspace grid
input_file=open("open_airspace_grid.dill", 'rb')
#input_file=open("open_airspace_grid_updated.dill", 'rb')##for 3d path planning
grid=dill.load(input_file)

##Initialise the flow control entity
graph=street_graph(G,edges,grid) 

# read line by line of file
flight_intention_list  = []
with open('flight_intention_big.csv') as file:
    for line in file:
        line = line.strip()
        line = line.split(',')
        flight_intention_list.append(line)

# path planning file TODO: match scenario name
path_plan_filename = 'Path_Planning_Big'

# Step 2: Generate traffic from the flight intention file
generated_traffic, loitering_edges_dict = bst.Intention2Traf(flight_intention_list, edges.copy())
print('Traffic generated!')

# create a loitering dill
output_file=open(f"loitering_edges.dill", 'wb')
dill.dump(loitering_edges_dict,output_file)
output_file.close()
print('Created loitering dill')

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
fig, ax = ox.plot_graph(G,node_color="w",show=False,close=False)


plan=None
# Step 3: Loop through traffic, find path, add to dictionary
scenario_dict = dict()
cnt=0
flight_plans_dict={}
sizes=[]
for flight in generated_traffic:
    cnt=cnt+1
    if cnt>20 :
        break #stop at 20 aircrafts or change that
        
    # First get the route and turns
    print(flight[0])
    origin = flight[3]
    destination = flight[4]

    priority = flight[7]
    aircraft_type = 1 if flight[1] == 'MP20' else 2
    
    if flight[0] in loitering_edges_dict.keys():
        plan = PathPlanning(aircraft_type,priority,grid,graph,gdf, origin[0], origin[1], destination[0], destination[1],True,loitering_edges_dict[flight[0]])
    else:
        plan = PathPlanning(aircraft_type,priority,grid,graph,gdf, origin[0], origin[1], destination[0], destination[1])
    flight_plans_dict[flight[0]]=plan
    route,turns,edges,next_turn,groups,in_constrained,turn_speed=plan.plan()
    if len(route)!=len(edges):
        print("unequal lens",len(route),len(edges))

    flight_plans_dict[flight[0]]=plan
    sizes.append(asizeof.asizeof(plan))



    
# =============================================================================
#     print("size",asizeof.asizeof(plan.graph))
#     print("size",asizeof.asizeof(plan.edge_gdf))
#     print("size",asizeof.asizeof(plan.os_keys_dict_pred))
# =============================================================================

# =============================================================================
# 
#     size=sys.getsizeof(plan.graph)+sys.getsizeof(plan.aircraft_type)+sys.getsizeof(plan.start_index_previous)+\
#     sys.getsizeof(plan.start_index)+sys.getsizeof(plan.start_in_open)+sys.getsizeof(plan.goal_index)
#     size=size+sys.getsizeof(plan.path)+sys.getsizeof(plan.os_keys_dict_pred)
#                                         
#     size=size+sys.getsizeof(plan.goal_index_next)+sys.getsizeof(plan.dest_in_open)+sys.getsizeof(plan.edge_gdf)+sys.getsizeof(plan.route)
#     size=size+sys.getsizeof(plan.turns)+sys.getsizeof(plan.priority)+ sys.getsizeof(plan.loitering)+sys.getsizeof(plan.in_same_cell)
#     size=size+sys.getsizeof(plan.init_succesful)+ \
#     sys.getsizeof(plan.start_point)+sys.getsizeof(plan.goal_point)+sys.getsizeof(plan.route)+ \
#     sys.getsizeof(plan.turns)+sys.getsizeof(plan.edges_list)+sys.getsizeof(plan.next_turn_point)+ \
#     sys.getsizeof(plan.groups)+sys.getsizeof(plan.in_constrained)+sys.getsizeof(plan.turn_speed)    
#     print("size ",size)
#     print(sys.getsizeof(plan.graph))
#     print(sys.getsizeof(plan.aircraft_type))
#     print(sys.getsizeof(plan.start_index_previous))
#     print(sys.getsizeof(plan.start_index))
#     print(sys.getsizeof(plan.start_in_open))
#     print(sys.getsizeof(plan.goal_index))
#     print(sys.getsizeof(plan.goal_index_next))
#     print(sys.getsizeof(plan.dest_in_open))
#     print(sys.getsizeof(plan.edge_gdf))
#     print(sys.getsizeof(plan.path))
#     print(sys.getsizeof(plan.os_keys_dict_pred))
#     print(sys.getsizeof(plan.route))
#     print(sys.getsizeof(plan.turns))
#     print(sys.getsizeof(plan.priority))
#     print(sys.getsizeof(plan.loitering))
#     print(sys.getsizeof(plan.in_same_cell))
#     print(sys.getsizeof(plan.init_succesful))
#     print(sys.getsizeof(plan.start_point))
#     print(sys.getsizeof(plan.goal_point))
#     print(sys.getsizeof(plan.route))
#     print(sys.getsizeof(plan.turns))
#     print(sys.getsizeof(plan.edges_list))
#     print(sys.getsizeof(plan.next_turn_point))
#     print(sys.getsizeof(plan.groups))
#     print(sys.getsizeof(plan.in_constrained))
#     print(sys.getsizeof(plan.turn_speed))
#         
# 
# =============================================================================


    if route!=[]:
        x_list=[]
        y_list=[]
            
        for r in route:
            x_list.append(r[0])
            y_list.append(r[1])
            
        ax.scatter(x_list,y_list, color='b')
        route = np.array(route)
        # Create dictionary
        scenario_dict[flight[0]] = dict()
        # Add start time
        scenario_dict[flight[0]]['start_time'] = flight[2]
        # Add aircraft type
        scenario_dict[flight[0]]['aircraft_type'] = 'M600'
        #Add lats
        scenario_dict[flight[0]]['lats'] = route[:,1]
        #Add lons
        scenario_dict[flight[0]]['lons'] = route[:,0]
        #Add turnbool
        scenario_dict[flight[0]]['turnbool'] = turns
        #Add alts
        scenario_dict[flight[0]]['alts'] = 0
        ## add start speed, If None then it is turnspeed
        # scenario_dict[flight[0]]['start_speed'] = flight[5]
        scenario_dict[flight[0]]['start_speed'] = None

        #Add active edges
        scenario_dict[flight[0]]['edges'] = edges
        #Add stroke group
        scenario_dict[flight[0]]['stroke_group'] = groups
        #Add next turn
        scenario_dict[flight[0]]['next_turn'] = next_turn
        #Add constarined airspace indicator
        scenario_dict[flight[0]]['airspace_type'] = in_constrained
        #add priority
        scenario_dict[flight[0]]['priority'] = flight[6]
        # add geoduration
        scenario_dict[flight[0]]['geoduration'] = flight[7]
        # add geocoords
        scenario_dict[flight[0]]['geocoords'] = flight[8]
    
    

print("size plan",asizeof.asizeof(sizes[0]))
print("size plan",asizeof.asizeof(sizes[1]))
print("size plan",asizeof.asizeof(sizes[2]))
print("size plan",asizeof.asizeof(sizes[3]))
print("size plan",asizeof.asizeof(sizes[4]))
print("size flow",asizeof.asizeof(graph))
l=[sizes,graph]
print("size sum",asizeof.asizeof(l))
    
    
print('All paths created!')
    
# Step 4: Create scenario file from dictionary
bst.Dict2Scn(r'Test_Scenario_Big.scn', 
              scenario_dict, path_plan_filename)

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
