import osmnx as ox
import pandas as pd
import json
import numpy as np
from airspace_design.entry_exit import entry_nodes, exit_nodes
import dill
from pyproj import Transformer,Proj, transform

# TODO: still need to correctly name the edge ids

def main():

    # read in data
    graph_path = 'whole_vienna/gis/finalized_graph.graphml'
    G = ox.io.load_graphml(graph_path)

    # read in airspace grid dill from parent directory
    with open('renamed_open_airspace_grid.dill', 'rb') as f:
        airspace_grid = dill.load(f)

    # get the nodes in the airspace grid
    grid = airspace_grid.grid

    # convert graph to geodataframe
    node_gdf,edge_gdf  = ox.graph_to_gdfs(G)


    # remove the key level from geodataframe index as it should all be equal to zero
    edge_gdf.reset_index(level=2, drop=True, inplace=True)

    # also remove some columns from dataframe
    edge_stroke = pd.DataFrame(edge_gdf['stroke_group'].copy())
    edge_flow = pd.DataFrame(edge_gdf['flow_group'].copy())

    # for each edge, get the stroke group
    stroke_array = np.sort(np.unique(edge_stroke.loc[:, 'stroke_group'].to_numpy()).astype(np.int64))
    flow_array = np.sort(np.unique(edge_gdf.loc[:, 'flow_group'].to_numpy()).astype(np.int64))

    stroke_length = {}
    for stroke in stroke_array:
        # get the edges with the stroke group
        stroke_edges = edge_gdf.loc[edge_gdf['stroke_group'] == str(stroke)].to_crs(crs='epsg:32633')
        
        # get the nodes of the edges
        length_stroke = sum(stroke_edges.length)

        # add to dictionary
        stroke_length[str(stroke)] = length_stroke

    # save node dictionary to JSON
    with open('airspace_design/stroke_length.json', 'w') as fp:
        json.dump(stroke_length, fp, indent=4)

    # for each edge, get the flow group
    flow_length = {}
    for flow in flow_array:
        # get the edges with the flow group
        flow_edges = edge_gdf.loc[edge_gdf['flow_group'] == str(flow)].to_crs(crs='epsg:32633')
        
        # get the nodes of the edges
        length_flow = sum(flow_edges.length)

        # add to dictionary
        flow_length[str(flow)] = length_flow

    # save node dictionary to JSON
    with open('airspace_design/flow_length.json', 'w') as fp:
        json.dump(flow_length, fp, indent=4)

    edge_gdf.drop(['oneway','length', 'bearing', 'edge_interior_angle', 'layer_allocation','bearing_diff', 'geometry'], axis=1, inplace=True)

    ############################ CREATE NODE DICTIONARY ################################

    # create a lat, lon and osmid list to make a dictionary of nodes
    lat_list = node_gdf['y'].tolist()
    lon_list = node_gdf['x'].tolist()
    osmid_list = node_gdf.index.values.tolist()

    # join lists into a dictionary with lat-lon as key and osmid as value
    node_dict = {}
    constrained_node_dict = {}
    for idx , _ in enumerate(lat_list):
        lat_lon = format(lat_list[idx], '.8f') + '-' + format(lon_list[idx], '.8f')
        osmid = osmid_list[idx]

        # dictionary of nodes with lat-lon as key and osmid as value for (streets)
        node_dict[lat_lon] = osmid

        # dictionary of nodes with osmid as key and lat-lon as value for (geofence)
        constrained_node_dict[int(osmid)] = (lat_list[idx], lon_list[idx])
    # save constrained node dictionary to JSON
    with open('airspace_design/constrained_node_dict.json', 'w') as fp:
        json.dump(constrained_node_dict, fp, indent=4)

    # add open airspace grid nodes to node dictionary
    transformer = Transformer.from_crs('epsg:32633','epsg:4326')
    
    open_airspace_nodes = []
    entry_edges = []
    exit_edges = []

    for node_info in grid:

        # get node osmid, lat and lon and save to dictionary
        node_id = node_info.key_index
        node_lat, node_lon = transformer.transform(node_info.center_x,node_info.center_y)
        lat_lon = format(node_lat, '.8f') + '-' + format(node_lon, '.8f')
        osmid = node_id
        node_dict[lat_lon] = osmid

        # save these nodes to a list
        open_airspace_nodes.append(node_id)

        # check if nodes have entry or exit and add to a list
        entry_nodes = node_info.entry_list
        exit_nodes = node_info.exit_list

        if entry_nodes:
            for node_entry in entry_nodes:
                entry_edges.append(f'{osmid}-{node_entry}')
        
        if exit_nodes:
            for node_exit in exit_nodes:
                exit_edges.append(f'{node_exit}-{osmid}')


    # add dummy node to dictionary
    dummy_osmid = 5000
    dummy_lat = 48.1351
    dummy_lon = 11.582

    node_dict[format(dummy_lat, '.8f') + '-' + format(dummy_lon, '.8f')] = int(dummy_osmid)

    # save node dictionary to JSON
    with open('airspace_design/nodes.json', 'w') as fp:
        json.dump(node_dict, fp, indent=4)

    ############################ CREATE EDGE DICTIONARY ################################

    # convert edges into a dictionary with the directionality as the key. 
    # and with the values as another sub dictionary, with stroke_group and layer_height
    edge_dict = edge_gdf.to_dict(orient='index')
    stroke_dict = edge_stroke.to_dict(orient='index')
    flow_dict = edge_flow.to_dict(orient='index')

    # create open airspace grid edges
    open_airspace_edges = [f'{dummy_osmid}-{open_node}' for open_node in open_airspace_nodes]

    # create a stroke group for open airspace edges
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
    
    # add open airspace edges to the edge dictionary
    for open_edge in open_airspace_edges:
        edge_dict_new[open_edge] = {'stroke_group': str(stroke_open), 'height_allocation': 'open'}
    
    # add open airspace edges to edge_dict
    edge_dict_new[f'{dummy_osmid}-{dummy_osmid}'] = {'stroke_group': str(stroke_open), 'height_allocation': 'open'}

    # save edge dictionary as json
    with open('airspace_design/edges.json', 'w') as fp:
        json.dump(edge_dict_new, fp, indent=4)

    # simplify edge_dict keys into a string rather than tuple for stroke
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
    for open_edge in open_airspace_edges:
        if str(stroke_open) in stroke_dict_new.keys():
            stroke_dict_new[str(stroke_open)].append(open_edge)
        else:
            stroke_dict_new[str(stroke_open)] = []
            stroke_dict_new[str(stroke_open)].append(open_edge)

    # save edge dictionary as json
    with open('airspace_design/strokes.json', 'w') as fp:
        json.dump(stroke_dict_new, fp, indent=4)
    
    # simplify edge_dict keys into a string rather than tuple for flow
    flow_dict_new = {}
    for key, value in flow_dict.items():
        new_key = value['flow_group']
        new_value = f'{key[0]}-{key[1]}'

        if new_key in flow_dict_new.keys():
            flow_dict_new[new_key].append(new_value)
        else:
            flow_dict_new[new_key] = []
            flow_dict_new[new_key].append(new_value)
            
    # save edge dictionary as json
    with open('airspace_design/flows.json', 'w') as fp:
        json.dump(flow_dict_new, fp, indent=4)
    
    # now create another json file containing information about dummies
    dummy_nodes = [int(dummy_osmid)] + open_airspace_nodes
    dummy_node_dict = {'osmid': dummy_nodes}
    
    dummy_edge_dict = {}
    # add entry and exit edges to dummy_edge_dict
    for entry_edge in entry_edges:
        dummy_edge_dict[entry_edge] = {'stroke_group': str(stroke_entry), 'height_allocation': 'open'}

    for exit_edge in exit_edges:
        dummy_edge_dict[exit_edge] = {'stroke_group': str(stroke_exit), 'height_allocation': 'open'}
    
    # add open edges to dummy_edge_dict
    for open_edge in open_airspace_edges:
        dummy_edge_dict[open_edge] = {'stroke_group': str(stroke_open), 'height_allocation': 'open'}
    
    dummy_stroke_dict = {}
    dummy_stroke_dict[str(stroke_entry)] = stroke_dict_new[str(stroke_entry)]
    dummy_stroke_dict[str(stroke_exit)] = stroke_dict_new[str(stroke_exit)]
    dummy_stroke_dict[str(stroke_open)] = stroke_dict_new[str(stroke_open)]

    # create dummy dict
    dummy_dict = {'nodes': dummy_node_dict, 'edges': dummy_edge_dict, 'strokes': dummy_stroke_dict}

    # save dummy dict as json
    with open('airspace_design/dummy.json', 'w') as fp:
        json.dump(dummy_dict, fp, indent=4)

if __name__ == '__main__':
    main()
