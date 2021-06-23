# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 12:40:07 2021

@author: nipat
"""

import osmnx
import pandas as pd
import pickle
from shapely.geometry import LineString

class osmnx_graph:
    def __init__(self):
        self._adj={}
        self._node={}
        self._pred={}
        self._succ={}
        self.edges=[]

G = osmnx.io.load_graphml(filepath='C:/Users/nipat/Downloads/M2_test_scenario-main/M2_test_scenario-main/graph_definition/gis/data/street_graph/processed_graph1.graphml')
edges_geometry=osmnx.graph_to_gdfs(G, nodes=False)["geometry"]



keys=edges_geometry.keys()

edges={}
tmp=0
for i in keys:
    if i[0]==tmp:
        #edges[i[0]][i[1]]=edges_geometry[i[0]][i[1]][0]
        edges[i[0]][i[1]]=list(edges_geometry[i[0]][i[1]][0].coords)
    else:
        tmp=i[0]
        edges[i[0]]={}
        #edges[i[0]][i[1]]=edges_geometry[i[0]][i[1]][0]
        edges[i[0]][i[1]]=list(edges_geometry[i[0]][i[1]][0].coords)
        
pickle.dump(edges, file = open("edges_geometry.pickle", "wb"))
    
g = osmnx.graph_to_gdfs(G)
with open('g.pickle', 'wb') as f:
    pickle.dump(g, f)
    
    
osG=osmnx_graph()
omsnx_keys_list=list(G._node.keys())
for i in omsnx_keys_list:
    osG._node[i]=G._node[i]
    osG._succ[i]=G._succ[i]
    osG._pred[i]=G._pred[i]
    osG._adj[i]=G._adj[i]
osG.edges=list(G.edges())

pickle.dump(osG, file = open("GG.pickle", "wb"))