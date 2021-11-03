import osmnx as ox
import geopandas as gpd
import json

def main():
    # read origin
    origin_locations = gpd.read_file("whole_vienna/gis/Sending_nodes.gpkg")
    
    # read graph
    lat_list = origin_locations['y_send'].tolist()
    lon_list = origin_locations['x_send'].tolist()

    lat_lon_origin = [(lat, lon) for lat, lon in zip(lat_list, lon_list)]

    # read destination
    destination_locations = gpd.read_file("whole_vienna/gis/Sending_nodes.gpkg")

    # read graph
    lat_list = destination_locations['y_send'].tolist()
    lon_list = destination_locations['x_send'].tolist()

    lat_lon_destination = [(lat, lon) for lat, lon in zip(lat_list, lon_list)]

    lat_lon_dict = {'origins': lat_lon_origin, 'destinations': lat_lon_destination}

    # save origin and destination to json
    with open('origin_destination.json', 'w') as fp:
        json.dump(lat_lon_dict, fp, indent=4)
    

if __name__ == '__main__':
    main()