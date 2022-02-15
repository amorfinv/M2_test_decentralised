# %%
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
from shapely.geometry import LineString
import geopandas as gpd
from multiprocessing import Pool as ThreadPool

# Initialize stuff
bst = BlueskySCNTools.BlueskySCNTools()

# Step 1: Import the graph we will be using
dir_path = os.path.dirname(os.path.realpath(__file__))
graph_path = dir_path.replace('current_code', 
          'current_code/whole_vienna/gis/finalized_graph.graphml')
G = ox.io.load_graphml(graph_path)
edges = ox.graph_to_gdfs(G)[1]
gdf=ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
print('Graph loaded!')

# load the geofence gdfs
gdf_path = dir_path.replace('current_code' , 'current_code/geofences.gpkg')
geo_gdf = gpd.read_file(gdf_path, driver='GPKG')

#Load the open airspace grid
input_file=open("open_airspace_final.dill", 'rb')
grid=dill.load(input_file)

##Initialise the flow control entity
graph=street_graph(G,edges,grid) 

# Create flow control dill
output_file=open(f"Flow_control.dill", 'wb')
dill.dump(graph,output_file)
output_file.close()    

def scenario_dills():
    dill_list = set()

    scenario_folder = 'scenarios/'
    scenario_folder_files = os.listdir(scenario_folder)
    scenario_folder_files = [file for file in scenario_folder_files if not 'R1' in file and not 'R2' in file and not 'R3' in file and not 'W1' in file and not 'W2' in file and not 'W3' in file and not 'batch' in file]

    def check_scen_dills(scen):
        with open(scenario_folder + scen, 'r') as f:
            lines = f.readlines()
            lines = lines[10:]
        for line in lines:
            dill_name = line.split(',')[2]
            dill_list.add(dill_name)


    for scenario in scenario_folder_files:
        check_scen_dills(scenario)


    return list(dill_list)

dill_list = scenario_dills()

# Origin-Destination pairs list
pairs_list = bst.pairs_list

# Create input array
input_arr = []
for i in dill_list:
    idx = int(i.split('_')[0])
    aircraft_type = i.split('_')[1]
    input_arr.append((idx, aircraft_type, pairs_list[idx]))

# %%

def create_dill(variables):
    file_num, aircraft_type, flight = variables
    # Step 3: Loop through traffic, find path, add to dictionary
    # create dills for two aircraft types
    # for idx_type, aircraft_type in enumerate(aircraft_types):
            
    # Break the loop at 10 files per aircraft type (just for now)
    # TODO: REMOVE THIS LATER WHEN CREATING ALL THE DILLS!
    # if file_num>10:#0 :
    #     break 
        #     break 
    #     break 

    # First get the origin, destinations
    origin_lon = flight[0]
    origin_lat = flight[1]

    destination_lon = flight[2]
    destination_lat = flight[3]
    
    route = []
    bigness_factor = 0

    if aircraft_type == 'MP20':
        actype = 1
    elif aircraft_type == 'MP30':
        actype = 2
    
    while not route:
        bigness_factor += 0.01
        # generate the path planning object
        plan = PathPlanning(actype, grid, graph,gdf, origin_lon, origin_lat, destination_lon, destination_lat, bigness_factor)
        
        # Check if route is empty and then recreate it
        # TODO: Convert to a while loop to find best sub graph
        #route,_,edges,_,_,_,_=plan.plan()
        route,turns,edges,next_turn,groups,in_constrained,turn_speed=plan.plan()
        
        if bigness_factor > 0.04:
            print(f'ERROR IN {file_num,flight}')
            break
    
    # if route==[]: ##TODO convert that to a while and 
    #     #No path was found so incease the exp_constant from default
    #     plan = PathPlanning(idx_type+1,grid,graph,gdf, origin_lon, origin_lat, destination_lon, destination_lat,0.03)
        
    #     #route,_,edges,_,_,_,_=plan.plan()
    #     route,turns,edges,next_turn,groups,in_constrained,turn_speed=plan.plan()
    
    # Check if route and edges are same size for quality of life purposes
    if len(route)!=len(edges):
        print("unequal lens",len(route),len(edges))
        print(flight)

    # check if route itnersects with the geofence
    # make a linestring from the coords
    linestring = LineString(route)

    # loop through the geofence gdfs
    for idx, geofence in enumerate(geo_gdf.geometry):
        # check if the linestring intersects with the geofence
        if linestring.intersects(geofence):
            print(f'{file_num} path interects with the geofence')
            
    # Delete the graph from plan and then save dill   
        # Delete the graph from plan and then save dill   
    # Delete the graph from plan and then save dill   
    del plan.flow_graph

    # save the dill!!
    file_loc_dill = f'{file_num}_{aircraft_type}'
    output_file=open(f"path_plan_dills/{file_loc_dill}.dill", 'wb')
    dill.dump(plan,output_file)
    output_file.close()
    # print(file_num, aircraft_type)
        
# def missingfiles(pairs_list, path):
#     '''Returns the missing dill file names.'''
#     # First of all, get all the files in given folder.
#     dill_files = os.listdir(path)
#     # Create empty list to store the missing files
#     missing = []
#     # for loop over pairs list
#     for i, pair in enumerate(pairs_list):
#         file_20 = f'{i}_MP20.dill'
#         file_30 = f'{i}_MP30.dill'
#         if file_20 not in dill_files or file_30 not in dill_files:
#             missing.append((i,pair))
#     return missing


def main():

    # create_dill(input_arr[0])
    pool = ThreadPool(32)
    results = pool.map(create_dill, input_arr)
    pool.close()
    
    # Make a while loop where we check for missing files. As long as we still get
    # missing files, repeat.
    # print('Checking for missing files.')
    # while True:
    #     missing_files = missingfiles(pairs_list, 'path_plan_dills')
    #     if not missing_files or len(missing_files) == 0:
    #         break
    #     print(f'Found {len(missing_files)} missing file(s).')
    #     # Create another pool, and create the missing files
    #     pool = ThreadPool(32)
    #     results = pool.map(create_dill, missing_files)
    #     pool.close()
    # print('Done.')

if __name__ == '__main__':
    main()
