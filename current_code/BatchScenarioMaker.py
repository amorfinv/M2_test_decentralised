
# %%
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:47:30 2021
@author: andub
"""
import osmnx as ox
import numpy as np
import BlueskySCNTools
from plugins.streets.flow_control import street_graph,bbox
from plugins.streets.agent_path_planning import PathPlanning,Path
from plugins.streets.open_airspace_grid import Cell, open_airspace
import os
import dill
import json
import sys
from pympler import asizeof
from multiprocessing import Pool as ThreadPool
from copy import deepcopy

# configurations
wind_speeds = [1,2,3] # knots
n_rogues = [1, 2, 3] # number of rogue aircrafts


# only select the files that contain _40_
scenario_fodler = 'scenarios/'
scenario_fodler_files = os.listdir(scenario_fodler)
scenario_files = [file for file in scenario_fodler_files if '_40_' in file]

# open the scenario files, copy the lines and add the rogue aircrafts
for scenario_file in scenario_files:

        # loop through the number of rogue aircrafts
        for n_rogue in n_rogues:

            # open the scenario file
            scenario_file_path = scenario_fodler + scenario_file
            with open(scenario_file_path) as file:
                rogue_lines = file.readlines()

            # loop through the numnber in n_rogue
            rogue_line = []
            for rogue in range(n_rogue):
                # add the rogue aircrafts after the 7th line
                rogue_line.append(f'00:00:00>SCHEDULE 00:15:00 PCALL rogues/R{rogue}.scn\n')
            
            # add the rogue lines after the 7th line
            rogue_lines[9:9] = rogue_line

            # write the lines to a new file
            scenario_file_path_new = 'rogue_scenarios/' + scenario_file.replace('.scn', f'_R{n_rogue}.scn')

            with open(scenario_file_path_new, 'w') as file:
                file.writelines(rogue_lines)
        
        # loop through the wind speeds
        for wind_speed in wind_speeds:

            # open the scenario file
            scenario_file_path = scenario_fodler + scenario_file
            with open(scenario_file_path) as file:
                wind_lines = file.readlines()
            
            # add wind speed
            wind_line = f'00:00:00>IMPL WINDSIM M2WIND\n00:00:00>SETM2WIND {wind_speed} 315\n'

            # add the rogue lines after the 7th line
            wind_lines[9:9] = wind_line
            
            # write the lines to a new file
            scenario_file_path_new = 'wind_scenarios/' + scenario_file.replace('.scn', f'_W{wind_speed}.scn')
            with open(scenario_file_path_new, 'w') as file:
                file.writelines(wind_lines)

# %%
        # # save the scenario file
        # scenario_file_path = scenario_fodler + scenario_file.replace('_40_',f'_40_rogues_{n_rogue}_')
        # with open(scenario_file_path, 'w') as file:
        #     file.writelines(lines)


# %%
