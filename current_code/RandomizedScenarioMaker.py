# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:47:30 2021
@author: andub

NOTE: If open airspace grid is empty then do not give 
origin and destination coordinates in open airspace
"""
from datetime import time
from networkx.drawing.nx_pylab import draw
import osmnx as ox
import numpy as np
import BlueskySCNTools
from plugins.streets.flow_control import street_graph,bbox
from plugins.streets.agent_path_planning import PathPlanning,Path
from plugins.streets.open_airspace_grid import Cell, open_airspace
import os
import dill
import json
import geopandas as gpd
from shapely.geometry import Point

###############################################################################
# Get smaller subset of destinations

## import a polygon of a smaller area
poly_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),'whole_vienna/gis/')
poly = gpd.read_file(poly_folder+'small_area.gpkg', driver='GPKG').to_crs(crs='epsg:4326')

## read in origin and destinations from origin_destination.json
with open('origin_destination.json') as f:
    origin_destination_dict = json.load(f)

origins_dict = origin_destination_dict['origins']
destinations_dict = origin_destination_dict['destinations']
center_dict = origin_destination_dict['center']

# make shapely points from origin destination, and center coordinates
origins = [Point([origin[1],origin[0]]) for _, origin in origins_dict.items()]
destinations = [Point([destination[1],destination[0]]) for _, destination in destinations_dict.items()]
centers = [Point([center[1],center[0]]) for _, center in center_dict.items()]

# only choose origins, destinations, and centers inside the polygon
origins_subset = [origin for origin in origins if poly.contains(origin).any()]
destinations_subset = [destination for destination in destinations if poly.contains(destination).any()]
centers_subset = [center for center in centers if poly.contains(center).any()]

# convert to list of coordinates from lon lat
origins_subset = [[origin.x for origin in origins_subset], [origin.y for origin in origins_subset]]
destinations_subset = [[destination.x for destination in destinations_subset], [destination.y for destination in destinations_subset]]
centers_subset = [[center.x for center in centers_subset], [center.y for center in centers_subset]]

# read in airspace structure
with open('airspace_design/layers.json', 'r') as filename:
    layer_dict = json.load(filename)

# get flight levels of airspace
flight_levels = layer_dict['info']['levels']

###############################################################################
# create a random scenario
scenario_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),'custom_scenarios')

test_scenario_name = 'random_scenario.scn'
path_plan_filename = 'random_scenario'
loitering_name = 'random_loitering'


# Initialize stuff
bst = BlueskySCNTools.BlueskySCNTools()

# Step 1: Import the graph we will be using
dir_path = os.path.dirname(os.path.realpath(__file__))
graph_path = dir_path.replace('current_code', 
          'current_code/whole_vienna/gis/layer_heights.graphml')
G = ox.io.load_graphml(graph_path)

edges = ox.graph_to_gdfs(G)[1]
gdf = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
print('Graph loaded!')


# Step 2: Generate random traffic
concurrent_ac = 4
aircraft_vel = 12 # [m/s]
max_time = 600 # [s]
dt = 10
min_dist = 1500 # [m]
ac_types = ['MP20', 'MP30']
priority_list = [1, 2, 3]

generated_traffic, loitering_edges_dict = bst.Graph2Traf(G, concurrent_ac, aircraft_vel, max_time, 
                                    dt, min_dist, origins_subset, destinations_subset, ac_types,
                                    priority_list)


#Load the open airspace grid
with open('open_airspace_grid.dill', 'rb') as filename:
    grid=dill.load(filename)

##Initialise the flow control entity
graph=street_graph(G,edges) 

# create a loitering dill
fpath_loitering = os.path.join(scenario_folder, loitering_name)
output_file=open(f"{fpath_loitering}.dill", 'wb')
dill.dump(loitering_edges_dict,output_file)
output_file.close()
print('Created loitering dill')

# Step 3: Loop through traffic, find path, add to dictionary
scenario_dict = dict()
flight_plans_dict={}
for flight in generated_traffic:
    # First get the route and turns
    origin = flight[3]
    destination = flight[4]

    priority = flight[7]
    aircraft_type = 1 if flight[1] == 'MP20' else 2
    
    if flight[0] in loitering_edges_dict.keys():
        plan = PathPlanning(aircraft_type,priority,grid,graph, gdf, origin[0], origin[1], destination[0], destination[1],True,loitering_edges_dict[flight[0]])
    else:
        plan = PathPlanning(aircraft_type,priority,grid,graph, gdf, origin[0], origin[1], destination[0], destination[1])
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
        scenario_dict[flight[0]]['alts'] = 0
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

print("Flight plans and search graphs saved!")
