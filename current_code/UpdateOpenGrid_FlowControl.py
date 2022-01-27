# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 16:18:34 2022

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
import math
from shapely.geometry import LineString
import shapely.geometry
from pyproj import  Transformer



#Load the open airspace grid
input_file=open("renamed_open_airspace_grid.dill", 'rb')
#input_file=open("open_airspace_grid_updated.dill", 'rb')##for 3d path planning
grid=dill.load(input_file)

for i in grid.grid:
    if 4481+216 in i.neighbors:
        i.neighbors.remove(4481+216)

    if 4481+139 in i.neighbors:
        i.neighbors.remove(4481+139) 

with open('airspace_border.geojson', 'r') as filename:
    airspace_border= json.load(filename)
f=airspace_border ["features"] [0]["geometry"]["coordinates"][0]      
     
transformer = Transformer.from_crs('epsg:4326','epsg:32633')
ff=[]
fff=[]
for i in range(len(f)-1):
    p1=transformer.transform(f[i][1],f[i][0])
    p2=transformer.transform(f[i+1][1],f[i+1][0])
    ff.append(((p1[0],p1[1]),(p2[0],p2[1])))
    fff.append((p1[0],p1[1]))
    
poly=shapely.geometry.MultiLineString(ff) 
poly1=shapely.geometry.Polygon(fff) 


 
a=1
for cell in grid.grid:

    line=LineString([cell.p0,cell.p1])
    aa=poly.intersection(line)

    if not aa.is_empty:

        if aa.type=="Point":

            p0=shapely.geometry.Point(cell.p0[0],cell.p0[1])
            b=poly1.contains(p0)
    
            if b:
                #cell.p1[0]=aa.x
                cell.p1[1]=aa.y

            else:
                #cell.p0[0]=aa.x
                cell.p0[1]=aa.y

        elif aa.type=="MultiPoint":
            if aa[0].y>aa[1].y:
                #cell.p0[0]=aa[0].x
                cell.p0[1]=aa[0].y            
                #cell.p1[0]=aa[1].x
                cell.p1[1]=aa[1].y
            else:
                #cell.p1[0]=aa[0].x
                cell.p1[1]=aa[0].y            
                #cell.p0[0]=aa[1].x
                cell.p0[1]=aa[1].y          
            
    line=LineString([cell.p1,cell.p2])

    aa=line.intersection(poly)

    if not aa.is_empty:

        if aa.type=="Point":
            p0=shapely.geometry.Point(cell.p2[0],cell.p2[1])
            b=poly1.contains(p0)

            if b:
                #cell.p1[0]=aa.x
                cell.p1[1]=aa.y
            else:
                #cell.p2[0]=aa.x
                cell.p2[1]=aa.y 
        elif aa.type=="MultiPoint":
            if aa[0].x>aa[1].x:
               # cell.p2[0]=aa[0].x
                cell.p2[1]=aa[0].y         
               # cell.p1[0]=aa[1].x
                cell.p1[1]=aa[1].y 
            else:
                #cell.p1[0]=aa[0].x
                cell.p1[1]=aa[0].y            
                #cell.p2[0]=aa[1].x
                cell.p2[1]=aa[1].y   
    line=LineString([cell.p2,cell.p3])
    aa=line.intersection(poly)

    if not aa.is_empty:
        if aa.type=="Point":
            p0=shapely.geometry.Point(cell.p2[0],cell.p2[1])
            b=poly1.contains(p0)
            if b:
                #cell.p3[0]=aa.x
                cell.p3[1]=aa.y
            else:
                #cell.p2[0]=aa.x
                cell.p2[1]=aa.y
        elif aa.type=="MultiPoint":
            if aa[0].y>aa[1].y:
                #cell.p3[0]=aa[0].x
                cell.p3[1]=aa[0].y            
                #cell.p2[0]=aa[1].x
                cell.p2[1]=aa[1].y
            else:
                #cell.p2[0]=aa[0].x
                cell.p2[1]=aa[0].y           
                #cell.p3[0]=aa[1].x
                cell.p3[1]=aa[1].y 
    line=LineString([cell.p3,cell.p0])
    aa=line.intersection(poly)

    if not aa.is_empty:
        if aa.type=="Point":
            p0=shapely.geometry.Point(cell.p0[0],cell.p0[1])
            b=poly1.contains(p0)
            if b:
                #cell.p3[0]=aa.x
                cell.p3[1]=aa.y
            else:
                #cell.p0[0]=aa.x
                cell.p0[1]=aa.y 
        elif aa.type=="MultiPoint":
            if aa[0].x>aa[1].x:
                #cell.p0[0]=aa[0].x
                cell.p0[1]=aa[0].y          
                #cell.p3[0]=aa[1].x
                cell.p3[1]=aa[1].y
            else:
                #cell.p3[0]=aa[0].x
                cell.p3[1]=aa[0].y           
                #cell.p0[0]=aa[1].x
                cell.p0[1]=aa[1].y 

    cell.center_x=(cell.p0[0]+cell.p1[0]+cell.p2[0]+cell.p3[0])/4
    cell.center_y=(cell.p0[1]+cell.p1[1]+cell.p2[1]+cell.p3[1])/4
    
    
grid.grid[222].p3[1]=5343690.012145453

cell=Cell()
cell.key_index=4705
cell.p0=[597278.7623781809,5344246.056715522]
cell.p1=[(597325.3279711378+597278.7623781809)/2,(5343729.813244521+5344246.056715522)/2]
cell.p2=[596933.1324057976,5344214.880602658]
cell.p3=[596933.1324057976,5344214.880602658]
cell.center_x=(cell.p0[0]+cell.p1[0]+cell.p2[0])/3
cell.center_y=(cell.p0[1]+cell.p1[1]+cell.p2[1])/3
cell.neighbors.append(4526)
cell.neighbors.append(4515)
grid.grid.append(cell)
grid.grid[4526-4481].neighbors.append(4705)
grid.grid[4515-4481].neighbors.append(4705)

cell=Cell()
cell.key_index=4706
cell.p0=[(9*595029.9318597122+595613.5655173562)/10,(9*5339182.516191929+5339194.525782009)/10]
cell.p1=cell.p0
cell.p2=[595029.9318597122,5339182.516191929]
cell.p3=[(2*595029.9318597122+595015.9519506333)/3,(2*5339182.516191929+5339861.90203396)/3]
cell.center_x=(cell.p0[0]+cell.p3[0]+cell.p2[0])/3
cell.center_y=(cell.p0[1]+cell.p3[1]+cell.p2[1])/3
cell.neighbors.append(4482)
cell.neighbors.append(4484)
grid.grid.append(cell)
grid.grid[4482-4481].neighbors.append(4706)
grid.grid[4484-4481].neighbors.append(4706)


output_file=open("open_airspace_final.dill", 'wb')
dill.dump(grid,output_file)
output_file.close()

# Step 1: Import the graph we will be using
dir_path = os.path.dirname(os.path.realpath(__file__))
graph_path = dir_path.replace('current_code', 
          'current_code/whole_vienna/gis/finalized_graph.graphml')
G = ox.io.load_graphml(graph_path)
#G = ox.io.load_graphml('processed_graph.graphml')
edges = ox.graph_to_gdfs(G)[1]
gdf=ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
print('Graph loaded!')

graph=street_graph(G,edges,grid) 

output_file=open(f"Flow_control.dill", 'wb')
dill.dump(graph,output_file)
output_file.close()