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

#Load the open airspace grid
input_file=open("renamed_open_airspace_grid.dill", 'rb')
grid=dill.load(input_file)

##Initialise the flow control entity
graph=street_graph(G,edges,grid) 

# Create flow control dill
output_file=open(f"Flow_control.dill", 'wb')
dill.dump(graph,output_file)
output_file.close()    

# Origin-Destination pairs list
pairs_list = bst.pairs_list

# Create input array
input_arr = []
for i, pair in enumerate(pairs_list):
    input_arr.append((i,pair))

def create_dill(variables):
    file_num, flight = variables
    # Step 3: Loop through traffic, find path, add to dictionary
    aircraft_types = ['MP20', 'MP30']
    # create dills for two aircraft types
    for idx_type, aircraft_type in enumerate(aircraft_types):
            
        # Break the loop at 10 files per aircraft type (just for now)
        # TODO: REMOVE THIS LATER WHEN CREATING ALL THE DILLS!
        # if file_num>10:#0 :
        #     break 

        # First get the origin, destinations
        origin_lon = flight[0]
        origin_lat = flight[1]

        destination_lon = flight[2]
        destination_lat = flight[3]
        
        route = []
        bigness_factor = 0
        
        while not route:
            bigness_factor += 0.01
            # generate the path planning object
            plan = PathPlanning(idx_type+1, grid, graph,gdf, origin_lon, origin_lat, destination_lon, destination_lat, bigness_factor)
            
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

        # Delete the graph from plan and then save dill   
        del plan.flow_graph

        # save the dill!!
        file_loc_dill = f'{file_num}_{aircraft_type}'
        output_file=open(f"path_plan_dills/{file_loc_dill}.dill", 'wb')
        dill.dump(plan,output_file)
        output_file.close()
        print(file_num, aircraft_type)

def main():
    # create_dill(input_arr[0])
    pool = ThreadPool(8)
    results = pool.map(create_dill, input_arr)
    pool.close()

if __name__ == '__main__':
    main()
