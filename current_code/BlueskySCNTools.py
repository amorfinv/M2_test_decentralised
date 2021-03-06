# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 09:50:37 2021

@author: andub
"""
import numpy as np
import random
import osmnx as ox
import networkx as nx
import os 
import json
import geopandas as gpd
#from pygeos.measurement import length
import rtree
from shapely.geometry import Polygon, Point
from shapely import affinity
from pyproj import Transformer, transformer

nm = 1852
ft = 1/0.3048

class BlueskySCNTools():
    def __init__(self):
        # create the pre generated pairs list
        self.PreGeneratedPaths()

        # Open strokes.JSON as a dictionary
        with open('airspace_design/strokes.json', 'r') as filename:
            self.stroke_dict = json.load(filename)
        
        # Opening edges.JSON as a dictionary
        with open('airspace_design/edges.json', 'r') as filename:
            self.edge_dict = json.load(filename)

        # Opening nodes.JSON as a dictionary
        with open('airspace_design/nodes.json', 'r') as filename:
            self.node_dict = json.load(filename)

        # Opening layers.JSON as a dictionary
        with open('airspace_design/layers.json', 'r') as filename:
            self.layer_dict = json.load(filename)
        
    def Drone2Scn(self, drone_id, aircraft_type, start_time, lats, lons, turnbool,alts = None, edges = None, group_num=None, next_turn = None, 
                cruise_speed_constraint = True, start_speed = None,in_constrained=True, priority = 1, geoduration = 0, geocoords = [],
                file_loc = ''):
        """Converts arrays to Bluesky scenario files. The first
        and last waypoints will be taken as the origin and 
        destination of the drone.
    
        Parameters
        ----------
        drone_id : str
            The ID of the drone to be created
        
        aircraft_type : str
            The type of aircraft to be created
            
        start_time : int [sec]
            The simulation time in seconds at which the drone starts its 
            journey.
            
        turn_speed : float [kts]
            The speed with which to turn at turn waypoints in knots.
            
        lats : float array/list [deg]
            The lattitudes of the waypoints.
            
        lons : float array/list [deg]
            The longitudes of the waypoints.
            
        turnbool : bool array/list
            True if waypoint is a turn waypoint, else false.
            
        alts : float array/list, optional [ft]
            Defines the required altitude at waypoints.
            
        edges : str
            Gives the edge IDs for the path.
            
        next_turn : int
            Gives when the next turn is going to be.

        cruise_speed_constraint : bool
            Choose if non-turn waypoints get a speed constraint.

        start_speed: float
            Set start speed for drone. If nothing is set, use the turn speed

        priority : int
            Gives the priority of the drone.

        """

        # Define the lines list to be returned
        lines = []
        
        # Speeds
        turn_speed = 10 # [kts]
        cruise_speed = 30 # [kts]
        speed_dist = 10 # [m]
        turn_dist = 10 # [m]

        # if cruise_speed_constraint:
        #     speeds, turnbool = self.TurnSpeedBuffer(lats, lons, turnbool, alts, 
        #                         turn_speed, cruise_speed, speed_dist, turn_dist)
        # else:
        #     speeds, turnbool = self.TurnSpeedBuffer(lats, lons, turnbool, alts, 
        #             turn_speed, '', speed_dist, turn_dist)
        # prep edges and next_turn
        active_edge = ['' if edge == -1 else f'{edge[0]}-{edge[1]}' for edge in edges]
        active_turns = ['' if turn_node == -1 else f'{turn_node[0]} {turn_node[1]}' for turn_node in next_turn]

        # find starting altitude
        if alts is None:
            height_type = self.edge_dict[active_edge[0]]['height_allocation']
            flight_levels = self.layer_dict['info']['levels']

            for flight_level in flight_levels:
                level_type = self.layer_dict['config'][height_type]['levels'][f'{flight_level}'][0]
                
                # start at lowest altitude cruise layer
                if level_type == 'C':

                    alts = f'{flight_level}'
                    break

        # First, define some strings we will often be using
        trn = f'ADDWPT {drone_id} FLYTURN\n'
        trn_spd = f'ADDWPT {drone_id} TURNSPEED {turn_speed}\n'
        fvr = f'ADDWPT {drone_id} FLYBY\n'
        lnav = f'LNAV {drone_id} ON\n'
        vnav = f'VNAV {drone_id} ON\n'
        # Convert start_time to Bluesky format
        start_time = round(start_time)
        m, s = divmod(start_time, 60)
        h, m = divmod(m, 60)
        start_time_txt = f'{h:02d}:{m:02d}:{s:02d}>'
        
        # Everyone starts at 25ft above ground
        # First, we need to create the drone, Matrice 600 going 30 kts for now.
        # Let's calculate its required heading.
        if not start_speed:
            start_speed = turn_speed
        
        qdr = self.qdrdist(lats[0], lons[0], lats[1], lons[1], 'qdr')

        if geocoords:
            cre_text = f'QUEUEM2 {drone_id},{aircraft_type},{file_loc},{lats[0]},{lons[0]},{lats[-1]},{lons[-1]},{qdr},{alts},{start_speed},{priority},{geoduration},{geocoords}\n'
        else:
            cre_text = f'QUEUEM2 {drone_id},{aircraft_type},{file_loc},{lats[0]},{lons[0]},{lats[-1]},{lons[-1]},{qdr},{alts},{start_speed},{priority},{geoduration},\n'
        lines.append(start_time_txt + cre_text)
        
        # # Then we need to for loop through all the lats
        # prev_wpt_turn = False
        # for i in range(1, len(lats)):
        #     if turnbool[i] == 1 or turnbool[i] == True:
        #         # We have a turn waypoint
        #         if prev_wpt_turn == False:
        #             # The previous waypoint was not a turn one, we need to enter
        #             # turn waypoint mode.
        #             lines.append(start_time_txt + trn)
        #             lines.append(start_time_txt + trn_spd)
        #     else:
        #         # Not a turn waypoint
        #         if prev_wpt_turn == True:
        #             # We had a turn waypoint initially, change to flyover mode
        #             lines.append(start_time_txt + fvr)
                    
        #     # Add the waypoint(TODO: fix this)
        #     speed = speeds[i]
        #     if any(alts):
        #         if speed:
        #             wpt_txt = f'ADDWPTM2 {drone_id} {lats[i]} {lons[i]} {alts[i]} {speed} {active_edge[i]} {group_num[i]} {active_turns[i]} {in_constrained[i]}\n'
        #         else:
        #             wpt_txt = f'ADDWPTM2 {drone_id} {lats[i]} {lons[i]} {alts[i]} ,, {active_edge[i]} {group_num[i]} {active_turns[i]} {in_constrained[i]}\n'

        #     else:
        #         if speed:
        #             wpt_txt = f'ADDWPTM2 {drone_id} {lats[i]} {lons[i]} ,, {speed} {active_edge[i]} {group_num[i]} {active_turns[i]} {in_constrained[i]}\n'
        #         else:
        #             wpt_txt = f'ADDWPTM2 {drone_id} {lats[i]} {lons[i]} ,,, {active_edge[i]} {group_num[i]} {active_turns[i]} {in_constrained[i]}\n'

        #     lines.append(start_time_txt + wpt_txt)
            
        #     # Set prev waypoint type value
        #     prev_wpt_turn = turnbool[i]
        
        # Delete aircraft at destination waypoint
        # if geocoords:
        #     lines.append(start_time_txt + f'{drone_id} ATDIST {lats[-1]} {lons[-1]} {5/nm} DELLOITER {drone_id}\n')
        # else:
        #     lines.append(start_time_txt + f'{drone_id} ATDIST {lats[-1]} {lons[-1]} {5/nm} DEL {drone_id}\n')
        
        # # Enable vnav and lnav
        # lines.append(start_time_txt + lnav)
        # lines.append(start_time_txt + vnav)

        return lines
    
    def Dict2Scn(self, filepath, dictionary, cruise_speed_constraint = True, start_speed = None):
        """Creates a scenario file from dictionary given that dictionary
        has the correct format.
    
        Parameters
        ----------
        filepath : str
            The file path and name of the scn file. 
            
        dictionary : dict
            This dictionary needs the format needed to use the Drone2Scn function.
            Drone_id is used as a main key, and then a sub dictionary is defined
            with the other variables.
            
            Example:
                dictionary = dict()
                dictionary['drone_id'] = dict()
                dictionary['drone_id']['start_time'] = start_time
                dictionary['drone_id']['lats'] = lats
                dictionary['drone_id']['lons'] = lons
                dictionary['drone_id']['turnbool'] = turnbool
                dictionary['drone_id']['alts'] = alts
                
            Set alts as None if no altitude constraints are needed.
        
        pathplanfilename: string
            This is the name of the dill containing path planning class used for flow
            control and replanning
        
        cruise_speed_constraint: bool
            Choose if non-turn waypoints get a speed constraint

        start_speed: float
            Set start speed for drone. If nothing is set, use the turn speed
    
        """
        if filepath[-4:] != '.scn':
            filepath = filepath + '.scn'
        
        with open(filepath, 'w+') as f:
            f.write('00:00:00>HOLD\n00:00:00>PAN 48.204011819028494 16.363471515762452\n00:00:00>ZOOM 15\n')
            f.write('00:00:00>ASAS ON\n00:00:00>RESO SPEEDBASEDV3\n00:00:00>CDMETHOD M2STATEBASED\n')
            f.write('00:00:00>STREETSENABLE\n')
            for drone_id in dictionary:
                try:
                    aircraft_type = dictionary[drone_id]['aircraft_type']
                    start_time = dictionary[drone_id]['start_time']
                    lats = dictionary[drone_id]['lats']
                    lons = dictionary[drone_id]['lons']
                    start_speed = dictionary[drone_id]['start_speed']
                    turnbool = dictionary[drone_id]['turnbool']
                    alts = dictionary[drone_id]['alts']
                    edges = dictionary[drone_id]['edges']
                    group_num = dictionary[drone_id]['stroke_group']
                    next_turn = dictionary[drone_id]['next_turn']
                    in_constrained = dictionary[drone_id]['airspace_type']
                    priority = dictionary[drone_id]['priority']
                    geoduration = dictionary[drone_id]['geoduration']
                    geocoords = dictionary[drone_id]['geocoords']
                    file_loc = dictionary[drone_id]['file_loc']
                except:
                    print('Key error. Make sure the dictionary is formatted correctly.')
                    return

                lines = self.Drone2Scn(drone_id, aircraft_type, start_time, lats, lons, turnbool, alts, edges, group_num, next_turn, 
                                      cruise_speed_constraint, start_speed, in_constrained, priority, geoduration, geocoords, file_loc)
                f.write(''.join(lines))

    def Intention2Traf(self, flight_intention_list, edges):
        """Processes a flight intention dataframe into traffic

        Args:
            flight_intention_list (list): [description]
        """
        # load the edges into an rtree
        edges_gdf = edges.copy()
        edge_dict = {}
        idx_tree = rtree.index.Index()
        i = 0
        for index, row in edges_gdf.iterrows():
            
            geom = row.loc['geometry']
            edge_dict[i] = (index[0], index[1])
            idx_tree.insert(i, geom.bounds)

            i += 1

        # read flight inention list to create trafgen list
        trafgen = []
        ac_no = 1

        loitering_edges_dict = {}

        for flight_intention in flight_intention_list:
          
            # get the starting time in seconds
            start_time = flight_intention[3][-8:]#.removeprefix('\ufeff')
            start_time = start_time.split(':')
            start_time = int(start_time[0])*3600 + int(start_time[1])*60 + int(start_time[2])

            # get aircraft type
            aircraft_type = flight_intention[2]
            
            # get last two entries of aicraft type for start_speed
            start_speed = float(aircraft_type[-2:])

            # get the origin location
            round_int = 10
            origin_lon = round(float(flight_intention[4][2:]),round_int)
            origin_lat = round(float(flight_intention[5][1:-2]),round_int)
            origin = (origin_lon, origin_lat)

            # get the destination location
            destination_lon = round(float(flight_intention[6][2:]),round_int)
            destination_lat = round(float(flight_intention[7][1:-2]),round_int)
            destination = (destination_lon, destination_lat)

            # get the priority
            priority = int(flight_intention[8])

            # get the geoduration
            geoduration = flight_intention[9]

            # find file location for the path planning
            file_loc = self.pairs_list.index((origin_lon,origin_lat,destination_lon,destination_lat))
            file_loc = str(file_loc) + '_' + aircraft_type

            # check if it has geocoords
            if flight_intention[10]:
                # get polygon coordianates from box and create into lat1 lon1 list
                box = flight_intention[10:]
                geocoords = box[2] + ' ' + box[0] + ' ' + box[3] + ' ' + box[0] + ' ' + box[3] + ' ' + box[1] + ' ' + box[2] + ' ' + box[1]
                
                # create shapely polygon
                poly = Polygon([Point(float(box[0]), float(box[2])), Point(float(box[0]), float(box[3])), Point(float(box[1]), float(box[3])), Point(float(box[1]), float(box[2]))])
                
                # scale the polygon in centroid of origin by 1.5*
                # poly = affinity.scale(poly, 1.5, 1.5, origin)

                # Add polygon to rtree
                nearest_trial = list(idx_tree.intersection(poly.bounds))

                list_intersecting_edges = [edge_dict[ii] for ii in nearest_trial]

                # fill the loitering edges dict
                loitering_edges_dict['D'+str(ac_no)] = list_intersecting_edges

                trafgen.append(('D'+str(ac_no), aircraft_type, start_time, origin, destination, file_loc ,start_speed, priority, geoduration, geocoords))
            else:
                trafgen.append(('D'+str(ac_no), aircraft_type, start_time, origin, destination, file_loc, start_speed, priority, 0, []))

            ac_no += 1
        
        return trafgen, loitering_edges_dict

    def TestIntention2Traf(self, flight_intention_list, edges):
        """Processes a custom flight intention list into traffic

        Args:
            flight_intention_list (list): [description]
        """
        # load the edges into an rtree
        edges_gdf = edges.copy()
        edge_dict = {}
        idx_tree = rtree.index.Index()
        i = 0
        for index, row in edges_gdf.iterrows():
            
            geom = row.loc['geometry']
            edge_dict[i] = (index[0], index[1])
            idx_tree.insert(i, geom.bounds)

            i += 1

        # read flight inention list to create trafgen list
        trafgen = []
        ac_no = 1

        loitering_edges_dict = {}

        for flight_intention in flight_intention_list:

            # get the starting time in seconds
            start_time = flight_intention['start_time']

            # get aircraft type
            aircraft_type = flight_intention['ac_type']

            # get the origin location
            origin_lon = flight_intention['origin'][1]
            origin_lat = flight_intention['origin'][0]
            origin = (origin_lon, origin_lat)

            # get the destination location
            destination_lon = flight_intention['destination'][1]
            destination_lat = flight_intention['destination'][0]
            destination = (destination_lon, destination_lat)

            # get the start speed
            start_speed = flight_intention['start_speed']

            # get the altitude
            altitude = flight_intention['altitude']

            # get the priority
            priority = flight_intention['priority']

            # get the geoduration
            geoduration = flight_intention['geoduration']
 
            # check if it has geocoords TODO: fix this
            if flight_intention['geocoords']:
                # get polygon coordianates from box and create into lat1 lon1 list
                box = flight_intention[10:]
                geocoords = box[2] + ' ' + box[0] + ' ' + box[3] + ' ' + box[0] + ' ' + box[3] + ' ' + box[1] + ' ' + box[2] + ' ' + box[1]
                
                # create shapely polygon
                poly = Polygon([Point(float(box[0]), float(box[2])), Point(float(box[0]), float(box[3])), Point(float(box[1]), float(box[3])), Point(float(box[1]), float(box[2]))])

                # Add polygon to rtree
                nearest_trial = list(idx_tree.intersection(poly.bounds))

                list_intersecting_edges = [edge_dict[ii] for ii in nearest_trial]

                # fill the loitering edges dict
                loitering_edges_dict['D'+str(ac_no)] = list_intersecting_edges

                trafgen.append(('D'+str(ac_no), aircraft_type, start_time, origin, destination, start_speed, altitude, priority, geoduration, geocoords))
            else:
                trafgen.append(('D'+str(ac_no), aircraft_type, start_time, origin, destination, start_speed, altitude, priority, 0, []))

            ac_no += 1
        
        return trafgen, loitering_edges_dict

    def Graph2Traf(self, G, concurrent_ac, aircraft_vel, max_time, dt, min_dist, 
                   orig_coords = None, dest_coords = None, ac_types = None, 
                   priority_list = None):
        """Creates random traffic using the nodes of graph G as origins and
        destinations.
    
        Parameters
        ----------
        G : graphml
            OSMNX graph, can be created using create_graph.py
            
        concurrent_ac : int
            The approximate number of aircraft flying at the same time.
            
        aircraft_vel : int/float [m/s]
            The approximate average velocity of aircraft
            
        max_time : int [s]
            The timespan for aircraft generation.
            
        dt : int [s]
            The time step to use. A smaller time step is faster but the number
            of concurrent aircraft will be less stable. 
            
        min_dist : int/float [m]
            The minimum distance a mission should have. This filters out the
            very short missions. 
            
        orig_coords : list
            List of lons and list of lats coordinates of origin locations.
            
        dest_coords : list
            List of lons and list of lats coordinates of destination locations.
        
        ac_types : list
            List of aircraft types.

        priority_list : list
            List of priorities.

        Output
        ------
        (ID, start_time, origin, destination, path_length)
        
        ID : str
            The flight ID.
            
        start_time : int [s]
            The simulation time at which the flight should start.
            
        origin : (lat,lon) [deg]
            The origin of the flight.
            
        destination : (lat,lon) [deg]
            The destination of the flight
            
        length : float [m]
            The approximate length of the path.
    
        """
        # get nearest nodes with osmnx
        orig_nodes = ox.nearest_nodes(G, orig_coords[0], orig_coords[1])
        dest_nodes = ox.nearest_nodes(G, dest_coords[0], dest_coords[1])
        
        # Some parameters
        timestep = 0
        ac_no = 1
        start_time = 0
        
        trafgen = []
        trafdist = np.empty((0,2))                   
        origins=orig_nodes.copy()
        destinations=dest_nodes.copy()
        
        # This loop is for time steps
        while start_time <= max_time:
            possible_origins = origins.copy()
            possible_destinations = destinations.copy()
            
            # We want to keep track of what aircraft might have reached their
            # destinations.
            # Skip first time step
            if timestep > 0:
                for aircraft in trafdist:
                    i = np.where(np.all(trafdist==aircraft,axis=1))[0]
                    # Subtract a dt*speed from length for each aircraft
                    dist = float(aircraft[1]) - aircraft_vel * dt
                    if dist < 0:
                        # Aircraft probably reached its destination
                        trafdist = np.delete(trafdist, i, 0)
                    else:
                        trafdist[i, 1] = dist
            
            # Amount of aircraft we need to add
            decrement_me = concurrent_ac - len(trafdist)
            # This loop is for each wave
            while decrement_me > 0:
                # Pick a random node from possible_origins
                idx_origin = random.randint(0, len(possible_origins)-1)
                orig_node = possible_origins[idx_origin]
                origin = (orig_coords[0][idx_origin], orig_coords[1][idx_origin])       

                # Do the same thing for destination
                idx_dest = random.randint(0, len(possible_destinations)-1)
                target_node = possible_destinations[idx_dest]
                destination = (dest_coords[0][idx_dest], dest_coords[1][idx_dest])

                # transformer
                transformer = Transformer.from_crs('epsg:4326','epsg:32633')
                origin_utm = transformer.transform(origin[1], origin[0])
                destination_utm = transformer.transform(destination[1], destination[0])

                
                # get euclidean distance betweem origin and destination
                length = np.sqrt((origin_utm[0]-destination_utm[0])**2 + (origin_utm[1]-destination_utm[1])**2)
                if length < min_dist:
                    # Distance is too short, try again                    
                    continue

                # Remove destinations and origins
                possible_origins.pop(idx_origin)
                possible_destinations.pop(idx_dest)

                # select and add aircraft type randomly
                aircraft_type = random.choice(ac_types)

                # start speed is last two entries of aircraft_type
                start_speed = int(aircraft_type[-2:])

                # select priority rand
                priority = random.choice(priority_list)

                # Append the new aircraft
                trafgen.append(('D'+str(ac_no), aircraft_type, start_time, origin, destination, start_speed, 30, priority, 0, []))

                # trafgen.append(('D'+str(ac_no), start_time, origin, destination, length))
                trafdist = np.vstack([trafdist, ['D'+str(ac_no),  length]])
                ac_no += 1
                decrement_me -= 1  
            # Go to the next time step
            timestep += 1
            start_time += dt
            
        return trafgen, {}

    def qdrdist(self, latd1, lond1, latd2, lond2, mode):
        """ Calculate bearing and distance, using WGS'84
            In:
                latd1,lond1 en latd2, lond2 [deg] :positions 1 & 2
            Out:
                qdr [deg] = heading from 1 to 2
                d [m]    = distance from 1 to 2 in m """
    
        # Haversine with average radius for direction
    
        # Check for hemisphere crossing,
        # when simple average would not work
    
        # res1 for same hemisphere
        res1 = self.rwgs84(0.5 * (latd1 + latd2))
    
        # res2 :different hemisphere
        a    = 6378137.0       # [m] Major semi-axis WGS-84
        r1   = self.rwgs84(latd1)
        r2   = self.rwgs84(latd2)
        res2 = 0.5 * (abs(latd1) * (r1 + a) + abs(latd2) * (r2 + a)) / \
            (np.maximum(0.000001,abs(latd1) + abs(latd2)))
    
        # Condition
        sw   = (latd1 * latd2 >= 0.)
    
        r    = sw * res1 + (1 - sw) * res2
    
        # Convert to radians
        lat1 = np.radians(latd1)
        lon1 = np.radians(lond1)
        lat2 = np.radians(latd2)
        lon2 = np.radians(lond2)
    
        
        #root = sin1 * sin1 + coslat1 * coslat2 * sin2 * sin2
        #d    =  2.0 * r * np.arctan2(np.sqrt(root) , np.sqrt(1.0 - root))
        # d =2.*r*np.arcsin(np.sqrt(sin1*sin1 + coslat1*coslat2*sin2*sin2))
    
        # Corrected to avoid "nan" at westward direction
        d = r*np.arccos(np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1) + \
                     np.sin(lat1)*np.sin(lat2))
    
        # Bearing from Ref. http://www.movable-type.co.uk/scripts/latlong.html
        coslat1 = np.cos(lat1)
        coslat2 = np.cos(lat2)
    
    
        qdr = np.degrees(np.arctan2(np.sin(lon2 - lon1) * coslat2,
            coslat1 * np.sin(lat2) - np.sin(lat1) * coslat2 * np.cos(lon2 - lon1)))
        
        if mode == 'qdr':
            return qdr
        elif mode == 'dist':
            return d
        else:
            return qdr, d
    
    def rwgs84(self, latd):
        """ Calculate the earths radius with WGS'84 geoid definition
            In:  lat [deg] (latitude)
            Out: R   [m]   (earth radius) """
        lat    = np.radians(latd)
        a      = 6378137.0       # [m] Major semi-axis WGS-84
        b      = 6356752.314245  # [m] Minor semi-axis WGS-84
        coslat = np.cos(lat)
        sinlat = np.sin(lat)
    
        an     = a * a * coslat
        bn     = b * b * sinlat
        ad     = a * coslat
        bd     = b * sinlat
    
        # Calculate radius in meters
        r = np.sqrt((an * an + bn * bn) / (ad * ad + bd * bd))
    
        return r
    
    def TurnSpeedBuffer(self, lats, lons, turnbool, alts, turnspeed, cruisespeed, speed_dist, turn_dist):
        """ Filters out waypoints that are very close to turn waypoints.
        

        Parameters
        ----------
        lats : array
            Waypoint latitudes
        lons : array
            Waypoint longitudes
        turnbool : bool array
            Whether waypoint is a turn waypoint or not.
        alts : array
            Altitude at waypoints.
        turnspeed : int [kts]
            The speed at which we are turning.
        cruisespeed : int[kts]
            The speed at which we are cruising.
        turndist : int [m]
            The buffer distance around a turn waypoint to filter for.

        Returns
        -------
        speeds : array
            The required speed at each waypoint.

        """
        # Number of waypoints
        num_wpts = len(lats)
        # Array that holds the speeds
        speeds = [cruisespeed] * num_wpts

        for i in range(num_wpts):
            if turnbool[i] == 0 or turnbool[i] == False:
                # We're only interested in turn waypoints
                continue
            # If we get here, it's a turn waypoint
            speeds[i] = turnspeed
            # What we want to do is check both future and previous waypoints
            # to see if they are too close to the turn waypoint.
            # First, let's check previous waypoints
            cumulative_distance = 0
            # Initialize the iterator
            j = i - 1
            while j >= 0:
                dist2wpt = self.qdrdist(lats[j], lons[j], lats[j+1], lons[j+1], 'dist')
                cumulative_distance += dist2wpt
                if cumulative_distance < turn_dist:
                    turnbool[j] = 1
                if cumulative_distance < speed_dist:
                    speeds[j] = turnspeed
                    j = j - 1
                else:
                    break
            
            # Check future waypoints
            cumulative_distance = 0
            # Initialize the iterator
            j = i + 1
            while j < num_wpts:
                dist2wpt = self.qdrdist(lats[j], lons[j], lats[j-1], lons[j-1], 'dist')
                cumulative_distance += dist2wpt
                if cumulative_distance < turn_dist:
                    turnbool[j] = 1
                if cumulative_distance < speed_dist:
                    speeds[j] = turnspeed
                    j = j + 1
                else:
                    break                
              
        return speeds, turnbool

    def PreGeneratedPaths(self):
        """ Generates a set of pre-generated paths for the aircraft. It reads the origins and 
        destinations and creates the list of tuples with
        (origin_lon, origin_lat, destination_lon, destination_lat)

        """
        origins = gpd.read_file('whole_vienna/gis/Sending_nodes.gpkg').to_numpy()[:,0:2]
        destinations = gpd.read_file('whole_vienna/gis/Recieving_nodes.gpkg').to_numpy()[:,0:2]

        pairs = []
        round_int = 10
        for origin in origins:
            for destination in destinations:
                if kwikdist(origin, destination) >=800:
                    lon1 = origin[0]
                    lat1 = origin[1]
                    lon2 = destination[0]
                    lat2 = destination[1]
                    pairs.append((round(lon1,round_int),round(lat1,round_int),round(lon2,round_int),round(lat2,round_int)))

        self.pairs_list = pairs


                
def kwikdist(origin, destination):
    """
    Quick and dirty dist [nm]
    In:
        lat/lon, lat/lon [deg]
    Out:
        dist [nm]
    """
    # We're getting these guys as strings
    lona = float(origin[0])
    lata = float(origin[1])

    lonb = float(destination[0])
    latb = float(destination[1])

    re      = 6371000.  # radius earth [m]
    dlat    = np.radians(latb - lata)
    dlon    = np.radians(((lonb - lona)+180)%360-180)
    cavelat = np.cos(np.radians(lata + latb) * 0.5)

    dangle  = np.sqrt(dlat * dlat + dlon * dlon * cavelat * cavelat)
    dist    = re * dangle
    return dist

# Testing here       
def main():
    bst = BlueskySCNTools()
    # Test dictionary
    dictionary = dict()
    dictionary['M1'] = dict()
    dictionary['M1']['start_time'] = 0
    dictionary['M1']['lats'] = [1,2,3,4,5]
    dictionary['M1']['lons'] = [1,2,3,4,5]
    dictionary['M1']['truebool'] = [False, True, False, True, False]
    dictionary['M1']['alts'] = [0,0,0,0,0]
    
    dictionary['M2'] = dict()
    dictionary['M2']['start_time'] = 548
    dictionary['M2']['lats'] = [5,4,3,2,1]
    dictionary['M2']['lons'] = [5,4,3,2,1]
    dictionary['M2']['truebool'] = [False, False, True, False, False]
    dictionary['M2']['alts'] = None
    
    bst.Dict2Scn('test', dictionary)
    
    # Also test trafgen
    dir_path = os.path.dirname(os.path.realpath(__file__))
    graph_path = dir_path.replace('Bluesky_tools', 
              'graph_definition/gis/data/street_graph/processed_graph.graphml')
    G = ox.io.load_graphml(graph_path)
    concurrent_ac = 10
    aircraft_vel = 13
    max_time = 400
    dt = 10
    min_dist = 1000
    
    trafgen = bst.Graph2Traf(G, concurrent_ac, aircraft_vel, max_time, dt, min_dist)
    print(trafgen)

if __name__ == '__main__':
    main()