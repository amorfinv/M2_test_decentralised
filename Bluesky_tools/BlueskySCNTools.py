# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 09:50:37 2021

@author: andub
"""

import numpy as np
import osmnx as ox

class BlueskySCNTools():
    def __init__(self):
        return

    def Drone2Scn(drone_id, start_time, turn_speed, lats, lons, turnbool, alts = None):
        """Converts arrays to Bluesky scenario files. The first
        and last waypoints will be taken as the origin and 
        destination of the drone.
    
        Parameters
        ----------
        drone_id : str
            The ID of the drone to be created
            
        start_time : int [sec]
            The simulation time in seconds at which the drone starts its 
            journey.
            
        turn_speed : float [kts]
            The speed with which to turn at turn waypoints in knots.
            
        lats : float array [deg]
            The lattitudes of the waypoints.
            
        lons : float array [deg]
            The longitudes of the waypoints.
            
        turnbool : bool array
            True if waypoint is a turn waypoint, else false.
            
        alts : float array, optional [ft]
            Defines the required altitude at waypoints.
    
        """
        
        # Define the lines list to be returned
        lines = []
        
        # First, define some strings we will often be using
        trn = f'ADDWPT {drone_id} FLYTURN\n'
        trn_spd = f'ADDWPT {drone_id} TURNSPEED {turn_speed}\n'
        fvr = f'ADDWPT {drone_id} FLYOVER\n'
        lnav = f'LNAV {drone_id} ON\n'
        vnav = f'VNAV {drone_id} ON\n'
        # Convert start_time to Bluesky format
        start_time = round(start_time)
        m, s = divmod(start_time, 60)
        h, m = divmod(m, 60)
        start_time_txt = f'{h:02d}:{m:02d}:{s:02d}>'
        
        # Everyone starts at 25ft above ground
        # First, we need to create the drone, Matrice 600 going 30 kts for now.
        cre_text = f'CRE {drone_id} M600 {lats[0]} {lons[0]} 25 30\n'
        lines.append(start_time_txt + cre_text)
        
        # Then we need to for loop through all the lats
        prev_wpt_turn = False
        for i in range(1, len(lats)):
            if turnbool[i] == True:
                # We have a turn waypoint
                if prev_wpt_turn == False:
                    # The previous waypoint was not a turn one, we need to enter
                    # turn waypoint mode.
                    lines.append(start_time_txt + trn)
                    lines.append(start_time_txt + trn_spd)
            else:
                # Not a turn waypoint
                if prev_wpt_turn == True:
                    # We had a turn waypoint initially, change to flyover mode
                    lines.append(start_time_txt + fvr)
                    
            # Add the waypoint
            if alts:
                wpt_txt = f'ADDWPT {drone_id} {lats[i]} {lons[i]} {alts[i]} 30\n'
            else:
                wpt_txt = f'ADDWPT {drone_id} {lats[i]} {lons[i]} ,, 30\n'
            lines.append(start_time_txt + wpt_txt)
            
            # Set prev waypoint type value
            prev_wpt_turn = turnbool[i]
            
        # Enable vnav and lnav
        lines.append(start_time_txt + vnav)
        lines.append(start_time_txt + lnav)

        return lines
    
# Testing here       
def main():
    bst = BlueskySCNTools
    lats = [1,2,3,4,5]
    lons = [1,2,3,4,5]
    turnbool = [False, False, True, False, False]
    lines = bst.Drone2Scn('M1', 0, 10, lats, lons, turnbool)
    with open('text.scn', 'w+') as f:
        for line in lines:
            f.write(line)
    return

if __name__ == '__main__':
    main()