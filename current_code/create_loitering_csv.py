"""
Code to get the geometry as a csv file for the loitering experiment.
"""
import osmnx as ox
import os
import pandas as pd
from shapely import wkt
import rtree


# Step 1: Import the graph we will be using
dir_path = os.path.dirname(os.path.realpath(__file__))
graph_path = dir_path.replace('current_code', 
          'current_code/whole_vienna/gis/layer_heights.graphml')
G = ox.io.load_graphml(graph_path)
edges = ox.graph_to_gdfs(G)[1]

# only take geometry column from csv
edges = edges[['geometry']]

# also drop the index "key"
edges = edges.reset_index(level=2, drop=True)

# save to a csv
edges.to_csv('edge_geometry.csv')


######### CODE BELOW GOES IN THE GEOFENCE PLUGIN OF BLUESKY ##############


# # convert csv to pandas dataframe
# edges = pd.read_csv('edge_geometry.csv')

# # convert geometry column to shapely geometry
# edges['geometry'] = edges['geometry'].apply(wkt.loads)

# # create rtree index for each edge
# edges_rtree = rtree.index.Index()
# edge_dict = {}
# for i, row in edges.iterrows():
#     edges_rtree.insert(i, row['geometry'].bounds)
#     edge_dict[i] = (row['u'], row['v'])
