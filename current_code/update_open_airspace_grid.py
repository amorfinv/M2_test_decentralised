# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 14:44:47 2021

@author: nipat
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
import geopandas


cc = geopandas.read_file("airspace_design/updated_constrained_airspace.gpkg")

constrained_poly=cc["geometry"][0].exterior._get_coords()

##Dill the flight_plans_dict
output_file=open("constrained_poly.dill", 'wb')
dill.dump(constrained_poly,output_file)
output_file.close()

# =============================================================================
# # Initialize stuff
# bst = BlueskySCNTools.BlueskySCNTools()
# 
# # Step 1: Import the graph we will be using
# dir_path = os.path.dirname(os.path.realpath(__file__))
# graph_path = dir_path.replace('current_code', 
#           'current_code/whole_vienna/gis/renamed_graph.graphml')
# G = ox.io.load_graphml(graph_path)
# #G = ox.io.load_graphml('processed_graph.graphml')
# edges = ox.graph_to_gdfs(G)[1]
# gdf=ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
# print('Graph loaded!')
# 
# 
# =============================================================================

# =============================================================================
# #Load the open airspace grid
# input_file=open("open_airspace_grid.dill", 'rb')
# #input_file=open("open_airspace_grid_updated.dill", 'rb')##for 3d path planning
# grid=dill.load(input_file)
# 
# 
# with open('old_to_new_nodes.json', 'r') as filename:
#     old_to_new_osmids = json.load(filename)
#     
# nodes_len=4481
# old_ids_new={}
# 
# for i,cell in enumerate(grid.grid):    
#     old_ids_new[cell.key_index]=4481+i
#     cell.key_index=4481+i
#     
# for i,cell in enumerate(grid.grid):
#     neigh=[]
#     for n in cell.neighbors:
#         neigh.append(old_ids_new[n])
#     cell.neighbors=neigh
#     exits=[]
#     for n in cell.exit_list:
#         exits.append(old_to_new_osmids[str(n)] )    
#     cell.exit_list=exits
#     entries=[]
#     for n in cell.entry_list:
#         entries.append(old_to_new_osmids[str(n)]) 
#     cell.entry_list=entries
#     
#     
#     
# ##Dill the flight_plans_dict
# output_file=open("renamed_open_airspace_grid.dill", 'wb')
# dill.dump(grid,output_file)
# output_file.close()
#     
# =============================================================================
