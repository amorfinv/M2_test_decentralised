import osmnx as ox
import pandas as pd
import json

graph_path = '../whole_vienna/gis/layer_heights.graphml'
G = ox.io.load_graphml(graph_path)

# convert graph to geodataframe
node_gdf,edge_gdf  = ox.graph_to_gdfs(G)

# remove the key level from geodataframe index as it should all be equal to zero
edge_gdf.reset_index(level=2, drop=True, inplace=True)

# also remove some columns from dataframe
edge_stroke = pd.DataFrame(edge_gdf['stroke_group'].copy())
edge_gdf.drop(['oneway','length', 'bearing', 'edge_interior_angle', 'layer_allocation','bearing_diff', 'geometry'], axis=1, inplace=True)

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
    json.dump(node_dict, fp, indent=4)

# convert edges into a dictionary with the directionality as the key. 
# and with the values as another sub dictionary, with stroke_group and layer_height
edge_dict = edge_gdf.to_dict(orient='index')
stroke_dict = edge_stroke.to_dict(orient='index')


# simplify edge_dict keys into a string rather than tuple
edge_dict_new = {}
for key, value in edge_dict.items():
    new_key = f'{key[0]}-{key[1]}'
    edge_dict_new[new_key] = value

# save edge dictionary as json
with open('edges.json', 'w') as fp:
    json.dump(edge_dict_new, fp, indent=4)

# simplify edge_dict keys into a string rather than tuple
stroke_dict_new = {}
for key, value in stroke_dict.items():
    new_key = value['stroke_group']
    new_value = f'{key[0]}-{key[1]}'

    if new_key in stroke_dict_new.keys():
        stroke_dict_new[new_key].append(new_value)
    else:
        stroke_dict_new[new_key] = []
        stroke_dict_new[new_key].append(new_value)


# save edge dictionary as json
with open('strokes.json', 'w') as fp:
    json.dump(stroke_dict_new, fp, indent=4)

# Opening edges.JSON as a dictionary
with open('edges.json', 'r') as filename:
    edge_dict = json.load(filename)

# Opening nodes.JSON as a dictionary
with open('nodes.json', 'r') as filename:
    node_dict = json.load(filename)
