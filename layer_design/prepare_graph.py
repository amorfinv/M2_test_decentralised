import osmnx as ox
import json

from functools import singledispatch

@singledispatch
def keys_to_strings(ob):
    return ob

@keys_to_strings.register
def _handle_dict(ob: dict):
    return {str(k): keys_to_strings(v) for k, v in ob.items()}

@keys_to_strings.register
def _handle_list(ob: list):
    return [keys_to_strings(v) for v in ob]

graph_path = '../graph_definition/gis/data/street_graph/processed_graph.graphml'
G = ox.io.load_graphml(graph_path)

# convert graph to geodataframe
g = ox.graph_to_gdfs(G)

# # get node and edge geodataframe
node_gdf = g[0]
edge_gdf = g[1]

# remove the key level from geodataframe index as it should all be equal to zero
edge_gdf.reset_index(level=2, drop=True, inplace=True)

# also remove some columns for dataframe
edge_gdf.drop(['osmid', 'lanes', 'name', 'highway', 'maxspeed', 'oneway', 'length', 'geometry', 
               'bearing', 'ref', 'edge_interior_angle'], axis=1, inplace=True)


# convert nodes into a list of osmids
node_list = list(node_gdf.index.values)

# convert edges into a dictionary with the directionality as the key. 
# and with the values as another sub dictionary, with stroke_group and layer_height
edge_dict = edge_gdf.to_dict(orient='index')

# correct some layer heights. TODO smart way to do this based on groups
edge_dict[(283324403, 655012)]['layer_height'] = 'height 2'
edge_dict[(655012, 33344801)]['layer_height'] = 'height 2'
edge_dict[(33344801, 33344802)]['layer_height'] = 'height 2'

# save nodes to a text file
with open('node_osmids.txt', 'w') as filehandle:
    for listitem in node_list:
        filehandle.write('%s\n' % listitem)

# save edge dictionary as json
with open('edges.json', 'w') as fp:
    json.dump(keys_to_strings(edge_dict), fp)

node_osmids = []

# open node_osmid.txt and save to a file
with open('node_osmids.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        node_osmid = line[:-1]

        # add item to the list
        node_osmids.append(node_osmid)

# Opening edges.JSON as a dictionary
with open('edges.json', 'r') as filename:
    edge_dict = json.load(filename)
