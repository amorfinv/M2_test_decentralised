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

first = True

def attr_bool():
    global first
    if first == True:
        first = False
        return [0]*46
    return array.array([random.randint(0,1) for i in range(46)])


orig_nodes_numbers = [30696015, 3155094143, 33345321,  25280685, 1119870220, 33302019,
          33144416, 378696, 33143911, 264055537, 33144706, 33144712, 
          33144719, 92739749]

dest_nodes_global = [291088171,  60957703, 30696019, 392251, 33301346, 26405238, 
              3963787755, 33345333, 378699, 33144821, 264061926, 33144695,
              33174086, 33345331]

# set directionality of groups with one edge
edge_directions = [(33302019, 378727, 0),   # group 0
                (33144416, 33144414, 0),    # group 1
                (30696015, 64975746, 0),    # group 2
                (378728, 33345331, 0),      # group 3
                (60631071, 401838, 0),      # group 4
                (264055540, 33144941, 0),   # group 5
                (33345285, 33345298, 0),    # group 6
                (251523470, 33345297, 0),   # group 7
                (33144366, 33143888, 0),    # group 8
                (1119870220, 394751, 0),    # group 9
                (378696, 3312560802, 0),    # group 10
                (64975949, 33345310, 0),    # group 11
                (3155094143, 64971266, 0),  # group 12
                (33144706, 33144601, 0),    # group 13
                (33344824, 33344825, 0),    # group 14
                (33344807, 33144550, 0),    # group 15
                (655012, 33144500, 0),      # group 16
                (33345286, 33345303, 0),    # group 17
                (283324403, 358517297, 0),  # group 18
                (33344802, 33344805, 0),    # group 19
                (264055537, 264055538, 0),  # group 20
                (29048460, 33345320, 0),    # group 21
                (33144712, 33144605, 0),    # group 22
                (33143911, 33143898, 0),    # group 23
                (29048469, 64972028, 0),    # group 24
                (64975551, 33345319, 0),    # group 25
                (92739749, 33144621, 0),    # group 26
                (33144633, 33144941, 0),    # group 27
                (33144560, 283324407, 0),   # group 28
                (25280685, 33344817, 0),    # group 29
                (33144566, 33144555, 0),    # group 30
                (33345332, 33345333, 0),    # group 31
                (33144471, 33144422, 0),    # group 32
                (33144659, 33144655, 0),    # group 33
                (33144719, 33144616, 0),    # group 34
                (33344808, 33144550, 0),    # group 35
                (33344812, 33344811, 0),    # group 36
                (245498401, 245498398, 0),  # group 37 
                (33144637, 320192043, 0),   # group 38 
                (33144755, 33144759, 0),    # group 39 
                (33344809, 2423479559, 0),  # group 40 
                (33344816, 392251, 0),      # group 41 
                (33345310, 33345289, 0),    # group 42 
                (33345299, 33344825, 0),    # group 43 
                (33345321, 33345291, 0),    # group 44
                (64975131, 60957703, 0)     # group 45 
                ]

################################## LOAD #####################################
dir_path = os.path.dirname(os.path.realpath(__file__))
graph_path = dir_path.replace('graph_definition',
'graph_definition/processed_graph.graphml')
G_init = ox.io.load_graphml(graph_path)

g = ox.graph_to_gdfs(G_init)
edges = g[1]
nodes = g[0]

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
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 46)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    # Make a copy of edge directions
    directions = copy.copy(edge_directions)
    dest_nodes = copy.copy(dest_nodes_global)
    # Change direction in function of bool_list
    for i in range(len(individual)):
        if individual[i] == 1 or individual[i] == True:
            direction = copy.copy(directions[i])
            directions[i] = (direction[1], direction[0], direction[2])
            
    # reoroder edge geodatframe
    edges_new = graph_funcs.set_direction(edges, directions)
    print(individual)
    G = ox.graph_from_gdfs(nodes, edges_new)
    
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