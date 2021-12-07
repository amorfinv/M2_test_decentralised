# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 18:45:30 2021

@author: nipat
"""

import numpy as np
from plugins.streets.flow_control import street_graph,bbox
from plugins.streets.agent_path_planning import PathPlanning,Path
from plugins.streets.open_airspace_grid import Cell, open_airspace
import os
import dill
import json
import sys
from pympler import asizeof




input_file=open("Path_Planning_Big.dill", 'rb')

path_dict=dill.load(input_file)

flow_control=path_dict[1]
flight_plans=path_dict[0]
plan1=flight_plans['D1']