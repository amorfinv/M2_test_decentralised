#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 11:14:29 2021

@author: andreibadea
"""

# Based on:
# https://github.com/DEAP/deap/blob/master/examples/ga/onemax.py

# Derek M Tishler
# Jul 2020

import array
import random
import sys

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import osmnx as ox
import networkx as nx
from os import path
from funcs import coins, graph_funcs
from evaluate_directionality import dijkstra_search_multiple
from whole_vienna.graph_builder import GraphBuilder
import random
import os
import copy

##Ray init code, user needs to apply#################
# see: https://docs.ray.io/en/master/walkthrough.html
import ray
from ray_map import ray_deap_map

class Node:
    def __init__(self,key,x,y):
        self.key=key
        self.x=x
        self.y=y
        self.children={}# each element of neigh is like [ox_key,edge_cost] 

def attr_bool():
    return array.array([random.randint(0,1) for i in range(len(GB.stroke_groups))])

GB = GraphBuilder()

orig_nodes_numbers = [30696015, 3155094143, 33345321,  25280685, 1119870220, 33302019,
          33144416, 378696, 33143911, 264055537, 33144706, 33144712, 
          33144719, 92739749]

dest_nodes_numbers = [291088171,  60957703, 30696019, 392251, 33301346, 26405238, 
              3963787755, 33345333, 378699, 33144821, 264061926, 33144695,
              33174086, 33345331]

ray.shutdown()
#ray.init(num_cpus=1) # will use default python map on current process, useful for debugging?
ray.init(num_cpus=4) # will batch out via ActorPool, slower vs above for trivial loads because overhead

'''
Eval is made arbitrarily more expensive to show difference. Tricky as DeltaPenalty skips evals sometimes.
'time python onemax_ray.py' on my machine(8 processors) shows:
num_cpus=1 (map): 25.5 sec(real)
num_cpus=2 (ray): 17.5 sec(real)
num_cpus=4 (ray): 13.0 sec(real)
num_cpus=7 (ray): 13.3 sec(real)
num_cpus=8 (ray): 13.6 sec(real)
'''
######################################################


##Example code updated, user needs to apply##########
def creator_setup():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", array.array, typecode='b', fitness = creator.FitnessMin)
# make sure to call locally
creator_setup()
######################################################


toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, 
                 toolbox.attr_bool, len(GB.stroke_groups))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    # Make a copy
    dest_nodes = copy.copy(dest_nodes_numbers)
            
    # reoroder edge geodatframe
    print(individual)
    G = GB.build_graph(individual)
    
    # Process nodes to put em in the right form
    omsnx_keys_list=list(G._node.keys())
    G_list=list(G._node)
    
    ###Initialise the graph for the search
    graph={}
    for i in range(len(omsnx_keys_list)):
        key=omsnx_keys_list[i]
        x=G._node[key]['x']
        y=G._node[key]['y']
        node=Node(key,x,y)
        children=list(G._succ[key].keys())
        for ch in children:
            cost=G[key][ch][0]['length']
            node.children[ch]=cost
        
        graph[key]=node
    orig_nodes = []
    for i, node in enumerate(orig_nodes_numbers):
        orig_nodes.append(graph[node])
    
    # Get cost
    cost = dijkstra_search_multiple(graph, orig_nodes, dest_nodes)
    return cost


toolbox.register("evaluate", evalOneMax)
# Here we apply a feasible constraint: 
# https://deap.readthedocs.io/en/master/tutorials/advanced/constraints.html
#toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 1.0, distance))
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

##This is different!#################################
toolbox.register("map", ray_deap_map, creator_setup = creator_setup)
######################################################

if __name__ == "__main__":
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, 
                        stats=stats, halloffame=hof)