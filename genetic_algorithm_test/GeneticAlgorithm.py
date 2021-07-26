# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 21:14:52 2021

@author: andub
"""

import osmnx as ox
import networkx as nx
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from os import path
import numpy as np
import graph_funcs_new
from evaluate_directionality import dijkstra_search_multiple
import random
import os
import copy

# Start genetic algorithm
creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
creator.create("Individual", list, fitness = creator.FitnessMin)

class Node:
    def __init__(self,key,x,y):
        self.key=key
        self.x=x
        self.y=y
        self.children={}# each element of neigh is like [ox_key,edge_cost] 

class GeneticAlgorithm:
    def __init__(self):
        self.orig_nodes_numbers = [393924, 33345303, 33143821, 33144572]
    
        self.dest_nodes = [33144484, 29048468, 33143995, 3956260281]
        self.edge_directions = [(393924, 33144487, 0),
                                (33345304, 29048468, 0),
                                (33144427, 3956260281, 0),
                                (25280482, 33143995, 0),
                                (33144487, 33144484, 0),]
    
    def GeneticAlgorithm(self):
        # set directionality of groups with one edge
        toolbox = base.Toolbox()
        
        toolbox.register("boollist", self.boollist)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.boollist, n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        toolbox.register("evaluate", self.CostFunction)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb = 0.05)
        
        # Do algorithm
        pop = toolbox.population(n = 50)
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
            
        CXPB, MUTPB = 0.5, 0.1
        
        fits = [ind.fitness.values[0] for ind in pop]
        
        g = 0
        
        while g<300:
            g = g + 1
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1[0], child2[0])
                    del child1.fitness.values
                    del child2.fitness.values
                    
            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant[0])
                    del mutant.fitness.values
                    
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                
            pop[:] = offspring
            fits = [ind.fitness.values[0] for ind in pop]
        
        best = pop[np.argmin([toolbox.evaluate(x) for x in pop])]
        return best
            
        
        
    def boollist(self):
        return [random.randint(0,1) for i in range(5)]
        
    def CostFunction(self, bool_list):
        ################################## LOAD #####################################
        dir_path = os.path.dirname(os.path.realpath(__file__))
        graph_path = dir_path.replace('genetic_algorithm_test',
            'genetic_algorithm_test/gis/processed_graph.graphml')
        G_init = ox.io.load_graphml(graph_path)
            
        g = ox.graph_to_gdfs(G_init)
        edges = g[1]
        nodes = g[0]
        # Make a copy of edge directions
        directions = copy.copy(self.edge_directions)
        dest_nodes = copy.copy(self.dest_nodes)
        # Change direction in function of bool_list
        for i in range(len(bool_list[0])):
            if bool_list[0][i] == 1 or bool_list[0][i] == True:
                direction = copy.copy(directions[i])
                directions[i] = (direction[1], direction[0], direction[2])
                
        # reoroder edge geodatframe
        edges_new = graph_funcs_new.set_direction(edges, directions)
        
        print(bool_list)
        
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
        for i, node in enumerate(self.orig_nodes_numbers):
            orig_nodes.append(graph[node])
        
        # Get cost
        cost = dijkstra_search_multiple(graph, orig_nodes, dest_nodes)
        return cost

def main():
    # Let's do genetics
    GA = GeneticAlgorithm()
    print(GA.CostFunction([[0,1,0,1,0]]))
    print(GA.CostFunction([[1,0,1,0,1]]))
    print(GA.CostFunction([[1,1,1,1,1]]))
    print(GA.CostFunction([[0,0,0,0,0]]))
    
    print('Best solution:', GA.GeneticAlgorithm())
    return

if __name__ == "__main__":
    main()
