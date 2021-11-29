# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 12:21:01 2021

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

from shapely import geometry



# Step 1: Import the graph we will be using
dir_path = os.path.dirname(os.path.realpath(__file__))
graph_path = dir_path.replace('current_code', 
          'current_code/whole_vienna/gis/layer_heights_simplified.graphml')
G = ox.io.load_graphml(graph_path)
#G = ox.io.load_graphml('processed_graph.graphml')
edges = ox.graph_to_gdfs(G)[1]
gdf=ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
print('Graph loaded!')

pt=geometry.Point(16.381609676387115,48.16806791791769)
bbox = ox.utils_geo.bbox_from_point((pt.y, pt.x), dist=300)
fig, ax = ox.plot_graph(G, bbox=bbox,show=False,close=False)
id1=1844621713
ax.scatter(G._node[id1]['x'],G._node[id1]['y'], color='r')
print(id1,G._pred[id1].keys())
id1=1844621714
ax.scatter(G._node[id1]['x'],G._node[id1]['y'], color='b')
print(id1,G._pred[id1].keys())
id1=1958117352
ax.scatter(G._node[id1]['x'],G._node[id1]['y'], color='g')
print(id1,G._pred[id1].keys())