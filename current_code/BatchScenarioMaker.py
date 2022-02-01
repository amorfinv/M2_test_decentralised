
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
scenario_folder = 'scenarios/'
scenario_folder_files = os.listdir(scenario_folder)
scenario_files = [file for file in scenario_folder_files if '_40_' in file]

# open the scenario files, copy the lines and add the rogue aircrafts
for scenario_file in scenario_files:

        # loop through the number of rogue aircrafts
        for n_rogue in n_rogues:

            # open the scenario file
            scenario_file_path = scenario_folder + scenario_file
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
            scenario_file_path = scenario_folder + scenario_file
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

# make a list of all regular scenario files
scenario_files = scenario_folder_files

# list of rogue scenarios
rogue_scenario_folder = 'rogue_scenarios/'
rogue_scenario_files = os.listdir(rogue_scenario_folder)

# list of wind scenarios
wind_scenario_folder = 'wind_scenarios/'
wind_scenario_files = os.listdir(wind_scenario_folder)

# combine all the lists
scenario_files = scenario_files + rogue_scenario_files + wind_scenario_files

# %%

very_low_scenarios = [file for file in scenario_files if 'very_low' in file]
low_scenarios = [file for file in scenario_files if 'low' in file and 'very' not in file]
medium_scenarios = [file for file in scenario_files if 'medium' in file]
high_scenarios = [file for file in scenario_files if 'high' in file]
ultra_scenarios = [file for file in scenario_files if 'ultra' in file]

# get number of files that have the word ultra, very_low, low, medium, high, very_high
middle_very_low = len(very_low_scenarios) // 2
middle_low = len(low_scenarios) // 2
middle_medium = len(medium_scenarios) // 2
middle_high = len(high_scenarios) // 2
middle_ultra = len(ultra_scenarios) // 2

# now divide the scenarios into two in middle
very_low_scenarios_1 = very_low_scenarios[:middle_very_low]
very_low_scenarios_2 = very_low_scenarios[middle_very_low:]

low_scenarios_1 = low_scenarios[:middle_low]
low_scenarios_2 = low_scenarios[middle_low:]

medium_scenarios_1 = medium_scenarios[:middle_medium]
medium_scenarios_2 = medium_scenarios[middle_medium:]

high_scenarios_1 = high_scenarios[:middle_high]
high_scenarios_2 = high_scenarios[middle_high:]

ultra_scenarios_1 = ultra_scenarios[:middle_ultra]
ultra_scenarios_2 = ultra_scenarios[middle_ultra:]

# assemble scenarios into two batches
batch_1 = very_low_scenarios_1 + low_scenarios_1 + medium_scenarios_1 + high_scenarios_1 + ultra_scenarios_1
batch_2 = very_low_scenarios_2 + low_scenarios_2 + medium_scenarios_2 + high_scenarios_2 + ultra_scenarios_2

# %%

# create the first batch scenatio

batch_1_scenario = []
for scenario in batch_1:
    # remove last 4 characters
    line1 = f'00:00:00>SCEN {scenario[:-4]}\n'
    line2 = f'00:00:00>PCALL m2/{scenario}\n'
    line3 = '00:00:00>FF\n'
    line4 = '00:00:00>SCHEDULE 01:30:00 HOLD\n'
    line5 = '00:00:00>SCHEDULE 01:30:00 DELETEALL\n\n'

    batch_1_scenario.append(line1)
    batch_1_scenario.append(line2)
    batch_1_scenario.append(line3)
    batch_1_scenario.append(line4)
    batch_1_scenario.append(line5)

# write to a file
with open('batch_scenarios/batch_1.scn', 'w') as file:
    file.writelines(batch_1_scenario)

# now do batch scenario 2
batch_2_scenario = []
for scenario in batch_2:
    # remove last 4 characters
    line1 = f'00:00:00>SCEN {scenario[:-4]}\n'
    line2 = f'00:00:00>PCALL m2/{scenario}\n'
    line3 = '00:00:00>FF\n'
    line4 = '00:00:00>SCHEDULE 01:30:00 HOLD\n'
    line5 = '00:00:00>SCHEDULE 01:30:00 DELETEALL\n\n'

    batch_2_scenario.append(line1)
    batch_2_scenario.append(line2)
    batch_2_scenario.append(line3)
    batch_2_scenario.append(line4)
    batch_2_scenario.append(line5)

# write to a file
with open('batch_scenarios/batch_2.scn', 'w') as file:
    file.writelines(batch_2_scenario)
# %%
