# %%
import osmnx as ox
import dill
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from shapely.affinity import scale
import numpy as np
import geopandas as gpd
import os
import BlueskySCNTools
from intersecting_ids import path_ids
from plugins.streets.agent_path_planning import PathPlanning
from plugins.streets.flow_control import street_graph
import pandas
from rich.pretty import pprint
from rich.progress import track
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich.live import Live
from rich.console import Group


# pprint(len(path_ids))

# Initialize stuff
bst = BlueskySCNTools.BlueskySCNTools()
pairs_list = bst.pairs_list
filtered_pairs_list = np.array(bst.pairs_list)[path_ids]

# load the geofence
dir_path = os.path.dirname(os.path.realpath(__file__))
gdf_path = dir_path.replace('current_code' , 'current_code/geofences.gpkg')
geo_gdf = gpd.read_file(gdf_path, driver='GPKG')

# %%
# read the graph
dir_path = os.path.dirname(os.path.realpath(__file__))
graph_path = dir_path.replace('current_code', 
          'current_code/whole_vienna/gis/finalized_graph.graphml')
G = ox.io.load_graphml(graph_path)
edges = ox.graph_to_gdfs(G)[1]
gdf=ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
pprint('Graph loaded!')

#Load the open airspace grid
input_file=open("open_airspace_final.dill", 'rb')
grid=dill.load(input_file)

##Initialise the flow control entity
graph=street_graph(G,edges,grid)


#Process to calculate max distance of intersecting paths
def kwikdist(lata, lona, latb, lonb):
    """
    Quick and dirty dist [nm]
    In:
        lat/lon, lat/lon [deg]
    Out:
        dist [m]
    """

    re      = 6371000.  # radius earth [m]
    dlat    = np.radians(latb - lata)
    dlon    = np.radians(((lonb - lona)+180)%360-180)
    cavelat = np.cos(np.radians(lata + latb) * 0.5)

    dangle  = np.sqrt(dlat * dlat + dlon * dlon * cavelat * cavelat)
    dist    = re * dangle

    return dist

# %%
# Step 3: Loop through traffic, find path, add to dictionary
table = Table(title="Intersecting Geofences")
table.add_column("Dill", style="cyan")
table.add_column("Geofence id", style="magenta")
table.add_column("Max intrusion distance", style="green")

progress = Progress()

progress_group = Group(table, progress)
# create dills for two aircraft types
min_distances = []
lines = []

with Live(progress_group, vertical_overflow="visible"):

    task1 = progress.add_task("[red]Processing...", total=len(filtered_pairs_list))

    for i, flight in enumerate(filtered_pairs_list):
    # for i in track(range(len(filtered_pairs_list)), "Processing..."):
        pprint(f'Processing flight {i}, {path_ids[i]}')
        # First get the origin, destinations
        origin_lon = flight[0]
        origin_lat = flight[1]

        destination_lon = flight[2]
        destination_lat = flight[3]
        
        route = []
        bigness_factor = 0
        
        while not route:
            bigness_factor += 0.01
            # generate the path planning object
            plan = PathPlanning(1, grid, graph,gdf, origin_lon, origin_lat, destination_lon, destination_lat, bigness_factor)
            
            route,turns,edges,next_turn,groups,in_constrained,turn_speed=plan.plan()
            
            if bigness_factor > 0.04:
                break
        
        # Check if route and edges are same size for quality of life purposes
        if len(route)!=len(edges):
            print("unequal lens",len(route),len(edges))

        # check if route itnersects with the geofence
        # make a linestring from the coords
        path = LineString(route)

        for idx, geofence in enumerate(geo_gdf.geometry):
            
            if path.intersects(geofence):

                # get boundary of geofence as a line string
                boundary_geo = geofence.boundary
                
                # get intersecting points of linestring
                intersections = boundary_geo.intersection(path)

                # make line string with intersecting point
                intersecting_line = LineString(intersections)

                # split intersecting line affine.scale..change factor
                split_locs = np.linspace(0,1,100)
                min_distance = []

                for split_loc in split_locs:
                    split_line = scale(intersecting_line,xfact=split_loc,yfact= split_loc,zfact=split_loc,origin=Point(intersecting_line.coords[0]))
                    split_point = Point(split_line.coords[1])
                    p1, p2 = nearest_points(split_point, boundary_geo)
                    min_distance.append(kwikdist(p1.x, p1.y, p2.x, p2.y))
                
                # select largest distance
                min_distances.append(max(min_distance))
                table.add_row(f'{path_ids[i]}', f'{idx}', f'{max(min_distance)}')
                progress.update(task1, advance=1)
                lines.append((f'{path_ids[i]}', f'{idx}', f'{max(min_distance)}'))

                # pprint((path_ids[i], idx, max(min_distance)))


print(f'maximum distance: {max(min_distances)}')                    

# write lines to file
import csv
with open('intersecting_geofences.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(lines)