import osmnx as ox
import pandas as pd
import json
import numpy as np
from entry_exit import entry_nodes, exit_nodes

def main():

    graph_path = '../whole_vienna/gis/layer_heights.graphml'
    G = ox.io.load_graphml(graph_path)

    # convert graph to geodataframe
    node_gdf,edge_gdf  = ox.graph_to_gdfs(G)

    # remove the key level from geodataframe index as it should all be equal to zero
    edge_gdf.reset_index(level=2, drop=True, inplace=True)

    # also remove some columns from dataframe
    edge_stroke = pd.DataFrame(edge_gdf['stroke_group'].copy())

    # for each edge, get the stroke group
    stroke_array = np.sort(np.unique(edge_stroke.loc[:, 'stroke_group'].to_numpy()).astype(np.int64))

    stroke_length = {}
    for stroke in stroke_array:
        # get the edges with the stroke group
        stroke_edges = edge_gdf.loc[edge_gdf['stroke_group'] == str(stroke)].to_crs(crs='epsg:32633')
        
        # get the nodes of the edges
        length_stroke = sum(stroke_edges.length)

        # add to dictionary
        stroke_length[str(stroke)] = length_stroke

    # save node dictionary to JSON
    with open('stroke_length.json', 'w') as fp:
        json.dump(stroke_length, fp, indent=4)


    edge_gdf.drop(['oneway','length', 'bearing', 'edge_interior_angle', 'layer_allocation','bearing_diff', 'geometry'], axis=1, inplace=True)

    # create a lat, lon and osmid list to make a dictionary of nodes
    lat_list = node_gdf['y'].tolist()
    lon_list = node_gdf['x'].tolist()
    osmid_list = node_gdf.index.values.tolist()

    # create a dummy node for open airspace
    dummy_osmid = node_gdf.index.values.max() + 1
    dummy_lat = 48.1351
    dummy_lon = 11.582

    # join lists into a dictionary with lat-lon as key and osmid as value
    node_dict = {}
    for idx , _ in enumerate(lat_list):
        lat_lon = format(lat_list[idx], '.8f') + '-' + format(lon_list[idx], '.8f')
        osmid = osmid_list[idx]

        node_dict[lat_lon] = osmid

    # add dummy node to dictionary
    node_dict[format(dummy_lat, '.8f') + '-' + format(dummy_lon, '.8f')] = int(dummy_osmid)

    # save node dictionary to JSON
    with open('nodes.json', 'w') as fp:
        json.dump(node_dict, fp, indent=4)

    # convert edges into a dictionary with the directionality as the key. 
    # and with the values as another sub dictionary, with stroke_group and layer_height
    edge_dict = edge_gdf.to_dict(orient='index')
    stroke_dict = edge_stroke.to_dict(orient='index')

    # create dummy edges from entry and exit nodes
    entry_edges = [f'{dummy_osmid}-{entry_node}' for entry_node in entry_nodes]
    exit_edges = [f'{exit_node}-{dummy_osmid}' for exit_node in exit_nodes]

    # create a stroke group for nodes travelling to constrained airspace
    stroke_entry = np.sort(np.unique(edge_stroke.loc[:, 'stroke_group'].to_numpy()).astype(np.int64)).max() + 1
    stroke_exit = stroke_entry + 1
    stroke_open = stroke_entry + 2

    # simplify edge_dict keys into a string rather than tuple
    edge_dict_new = {}
    for key, value in edge_dict.items():
        new_key = f'{key[0]}-{key[1]}'
        edge_dict_new[new_key] = value

    # add entry and exit edges to edge_dict
    for entry_edge in entry_edges:
        edge_dict_new[entry_edge] = {'stroke_group': str(stroke_entry), 'height_allocation': 'open'}

    for exit_edge in exit_edges:
        edge_dict_new[exit_edge] = {'stroke_group': str(stroke_exit), 'height_allocation': 'open'}
    
    # add open airspace edges to edge_dict
    edge_dict_new[f'{dummy_osmid}-{dummy_osmid}'] = {'stroke_group': str(stroke_open), 'height_allocation': 'open'}

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

    # add entry and exit edges to stroke_dict
    for entry_edge in entry_edges:
        if str(stroke_entry) in stroke_dict_new.keys():
            stroke_dict_new[str(stroke_entry)].append(entry_edge)
        else:
            stroke_dict_new[str(stroke_entry)] = []
            stroke_dict_new[str(stroke_entry)].append(entry_edge)

    for exit_edge in exit_edges:
        if str(stroke_exit) in stroke_dict_new.keys():
            stroke_dict_new[str(stroke_exit)].append(exit_edge)
        else:
            stroke_dict_new[str(stroke_exit)] = []
            stroke_dict_new[str(stroke_exit)].append(exit_edge)

    # add open airspace edges to stroke_dict
    stroke_dict_new[str(stroke_open)] = []
    stroke_dict_new[str(stroke_open)].append(f'{dummy_osmid}-{dummy_osmid}')

    # save edge dictionary as json
    with open('strokes.json', 'w') as fp:
        json.dump(stroke_dict_new, fp, indent=4)
    
    # now create another json file containing information about dummies
    dummy_node_dict = {'osmid': int(dummy_osmid)}
    
    dummy_edge_dict = {}
    # add entry and exit edges to dummy_edge_dict
    for entry_edge in entry_edges:
        dummy_edge_dict[entry_edge] = {'stroke_group': str(stroke_entry), 'height_allocation': 'open'}

    for exit_edge in exit_edges:
        dummy_edge_dict[exit_edge] = {'stroke_group': str(stroke_exit), 'height_allocation': 'open'}
    
    # add open airspace edges to dummy_edge_dict
    dummy_edge_dict[f'{dummy_osmid}-{dummy_osmid}'] = {'stroke_group': str(stroke_open), 'height_allocation': 'open'}
    
    dummy_stroke_dict = {}
    dummy_stroke_dict[str(stroke_entry)] = stroke_dict_new[str(stroke_entry)]
    dummy_stroke_dict[str(stroke_exit)] = stroke_dict_new[str(stroke_exit)]
    dummy_stroke_dict[str(stroke_open)] = stroke_dict_new[str(stroke_open)]

    # create dummy dict
    dummy_dict = {'nodes': dummy_node_dict, 'edges': dummy_edge_dict, 'strokes': dummy_stroke_dict}

    # save dummy dict as json
    with open('dummy.json', 'w') as fp:
        json.dump(dummy_dict, fp, indent=4)

if __name__ == '__main__':
    main()
