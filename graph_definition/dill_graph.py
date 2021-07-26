# -*- coding: utf-8 -*-

"""

Created on Wed Jun 23 12:40:07 2021



@author: nipat

"""



import osmnx
import os
import dill


dir_path = os.path.dirname(os.path.realpath(__file__))
graph_path = dir_path.replace('graph_definition',
    'graph_definition/gis/data/street_graph/processed_graph.graphml')


G = osmnx.io.load_graphml(filepath=graph_path)

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

dill.dump(edges, file = open("edges_geometry.dill", "wb"))

g = osmnx.graph_to_gdfs(G)

edge_gdf = g[1]



dill.dump(edge_gdf, file = open("edge_gdf.dill", "wb"))

dill.dump(G, file = open("G-multigraph.dill", "wb"))
