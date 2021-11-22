import osmnx as ox
import geopandas as gpd
import json

def main():
    # read origin
    origin_locations = gpd.read_file("whole_vienna/gis/Sending_nodes.gpkg")

    # read graph
    index_origin = list(origin_locations.index.values)
    index_origin = [int(i) for i in index_origin]

    lat_list = origin_locations['y_send'].tolist()
    lon_list = origin_locations['x_send'].tolist()

    lat_lon_origin = [(lat, lon) for lat, lon in zip(lat_list, lon_list)]

    # create a dictionary with the key as the index and value as the lat lon
    origin_dict = dict(zip(index_origin, lat_lon_origin))

    # read destination
    destination_locations = gpd.read_file("whole_vienna/gis/Sending_nodes.gpkg")

    # read graph
    index_destination = list(destination_locations.index.values)
    index_destination = [int(i) for i in index_destination]
    
    lat_list = destination_locations['y_send'].tolist()
    lon_list = destination_locations['x_send'].tolist()

    lat_lon_destination = [(lat, lon) for lat, lon in zip(lat_list, lon_list)]
    destination_dict = dict(zip(index_destination, lat_lon_destination))

    # read center
    center_locations = gpd.read_file("whole_vienna/gis/center_points.gpkg")
    center_locations = center_locations.to_crs(epsg=4326)
    
    # read graph
    index_center = list(center_locations['id'])

    lat_list = center_locations['geometry'].y.tolist()
    lon_list = center_locations['geometry'].x.tolist()
    
    lat_lon_center = [(lat, lon) for lat, lon in zip(lat_list, lon_list)]
    center_dict = dict(zip(index_center, lat_lon_center))

    lat_lon_dict = {'origins': origin_dict, 'destinations': destination_dict, 'center': center_dict}
    
    # save origin and destination to json
    with open('origin_destination.json', 'w') as fp:
        json.dump(lat_lon_dict, fp, indent=4)
    

if __name__ == '__main__':
    main()