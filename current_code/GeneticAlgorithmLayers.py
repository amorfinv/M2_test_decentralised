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

import random

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import pandas

import osmnx as ox
from os import path
from whole_vienna.height_cost_function import init_height_cost_estimate, cost_estimate


##Ray init code, user needs to apply#################
# see: https://docs.ray.io/en/master/walkthrough.html
import ray
from ray_map import ray_deap_map

# working path
gis_data_path = 'whole_vienna/gis'

# Load MultiDigraph from create_graph.py
G = ox.load_graphml(filepath=path.join(gis_data_path, '', 'prep_height_allocation_2.graphml'))

# convert to gdf
nodes, edges = ox.graph_to_gdfs(G)

# initiaize cost estimate
init_genome, group_dict, node_connectivity, stroke_lenghts = init_height_cost_estimate(nodes, edges)


ray.shutdown()
ray.init(num_cpus=1) # will use default python map on current process, useful for debugging?
#ray.init(num_cpus=1) # will batch out via ActorPool, slower vs above for trivial loads because overhead

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
    creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
    creator.create("Individual", list , fitness = creator.FitnessMin)
# make sure to call locally
creator_setup()
######################################################


toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, 
                 toolbox.attr_bool, len(stroke_lenghts))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    # Get cost
    total_cost = cost_estimate(individual, group_dict, node_connectivity, stroke_lenghts)
    print('--------------------------------------')
    print(individual)
    print(f'Cost for this individual: {total_cost}')
    return total_cost


toolbox.register("evaluate", evalOneMax)
# Here we apply a feasible constraint: 
# https://deap.readthedocs.io/en/master/tutorials/advanced/constraints.html
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

##This is different!#################################
toolbox.register("map", ray_deap_map, creator_setup = creator_setup)
######################################################

if __name__ == "__main__":
    # First, let's select 10 random numbers from which we start the optimisation
    seeds = [5, 4552709,9107886,  3397946, 270093, 8583586, 6090281, 2776495, 4926941, 465159]
    for i, seed in enumerate(seeds):
        random.seed(seed)
        pop = toolbox.population(n=6)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
    
        population, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, 
                            stats=stats, halloffame=hof)

        # Save progression to CSV
        df_log = pandas.DataFrame(log)
        df_log.to_csv(f'gendata/min{i+1}.csv', index=False)
        # Save individual to txt
        with open(f'gendata/min{i+1}.txt', 'w') as f:
            for individual in hof.items:
                f.write(str(individual))
                f.write('\n')
        
        # Shutdown at the end
        ray.shutdown()