import osmnx as ox
import json

graph_path = '../graph_definition/gis/data/street_graph/processed_graph.graphml'
G = ox.io.load_graphml(graph_path)

# convert graph to geodataframe
g = ox.graph_to_gdfs(G)

# # get node and edge geodataframe
node_gdf = g[0]
edge_gdf = g[1]

# remove the key level from geodataframe index as it should all be equal to zero
edge_gdf.reset_index(level=2, drop=True, inplace=True)

# also remove some columns from dataframe
edge_gdf.drop(['osmid', 'lanes', 'name', 'highway', 'maxspeed', 'oneway', 'length', 'geometry', 
               'bearing', 'ref', 'edge_interior_angle'], axis=1, inplace=True)

node_gdf.drop(['street_count', 'highway', 'geometry'], axis=1, inplace=True)

# create a lat, lon and osmid list to make a dictionary of nodes
lat_list = node_gdf['y'].tolist()
lon_list = node_gdf['x'].tolist()
osmid_list = node_gdf.index.values.tolist()

# join lists into a dictionary with lat-lon as key and osmid as value
lat_lon_list = []
node_dict = {}
for idx , _ in enumerate(lat_list):
    lat_lon = format(lat_list[idx], '.8f') + '-' + format(lon_list[idx], '.8f')
    osmid = osmid_list[idx]

    node_dict[lat_lon] = osmid

# save node dictionary to JSON
with open('nodes.json', 'w') as fp:
    json.dump(node_dict, fp)

# convert edges into a dictionary with the directionality as the key. 
# and with the values as another sub dictionary, with stroke_group and layer_height
edge_dict = edge_gdf.to_dict(orient='index')

# correct some layer heights. TODO smart way to do this based on groups
edge_dict[(283324403, 655012)]['layer_height'] = 'height 2'
edge_dict[(655012, 33344801)]['layer_height'] = 'height 2'
edge_dict[(33344801, 33344802)]['layer_height'] = 'height 2'

# simplify edge_dict keys into a string rather than tuple
edge_dict_new = {}
for key, value in edge_dict.items():
    new_key = f'{key[0]}-{key[1]}'
    edge_dict_new[new_key] = value

# save edge dictionary as json
with open('edges.json', 'w') as fp:
    json.dump(edge_dict_new, fp)

# Opening edges.JSON as a dictionary
with open('edges.json', 'r') as filename:
    edge_dict = json.load(filename)

# Opening nodes.JSON as a dictionary
with open('nodes.json', 'r') as filename:
    node_dict = json.load(filename)
