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
from funcs import coins, graph_funcs
from evaluate_directionality import dijkstra_search_multiple
import random

class GeneticAlgorithm:
    def GeneticAlgorithm(self):
        self.orig_nodes = [30696015, 3155094143, 33345321,  25280685, 1119870220, 33302019,
                  33144416, 378696, 33143911, 264055537, 33144706, 33144712, 
                  33144719, 92739749]
    
        self.dest_nodes = [291088171,  60957703, 30696019, 392251, 33301346, 26405238, 
                      3963787755, 33345333, 378699, 33144821, 264061926, 33144695,
                      33174086, 33345331]
        
        ################################## CREATE GRAPH #####################################
        # working paths
        gis_data_path = path.join('gis','data')
    
        # convert shapefile to shapely polygon
        center_poly = graph_funcs.poly_shapefile_to_shapely(path.join(gis_data_path, 'street_info', 'new_poly_shapefile.shp'))
    
        # create MultiDigraph from polygon
        G = ox.graph_from_polygon(center_poly, network_type='drive', simplify=True)
    
        # save as osmnx graph
        ox.save_graphml(G, filepath=path.join(gis_data_path, 'street_graph', 'raw_streets.graphml'))
        ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'street_graph', 'raw_graph.gpkg'))
    
        # remove unconnected streets and add edge bearing attrbute 
        G = graph_funcs.remove_unconnected_streets(G)
        ox.add_edge_bearings(G)
        
        #ox.plot.plot_graph(G)
    
        # manually remove some nodes and edges to clean up graph
        nodes_to_remove = [33144419, 33182073, 33344823, 83345330, 33345337,
                           33345330, 33344804, 30696018, 5316966255, 5316966252, 33344821,
                           2335589819, 245498397, 287914700, 271016303, 2451285012, 393097]
        edges_to_remove = [(291088171, 3155094143), (60957703, 287914700), (2451285012, 287914700),
                           (25280685, 30696019), (30696019, 25280685), (392251, 25280685), 
                           (25280685, 392251), (33301346, 1119870220),  
                           (33345331, 33345333), (378699, 378696), (378696, 33143911), 
                           (33143911, 33144821), (264061926, 264055537), (33144706, 33144712),
                           (33144712, 33174086), (33174086, 33144719), (33144719, 92739749),
                           (33345319, 29048469), (287914700, 60957703), (213287623, 251207325),
                           (251207325, 213287623)]
        G.remove_nodes_from(nodes_to_remove)
        G.remove_edges_from(edges_to_remove)
        
        #ox.plot.plot_graph(G)
    
        # convert graph to geodataframe
        g = ox.graph_to_gdfs(G)
    
        # # get node and edge geodataframe
        self.nodes = g[0]
        edges = g[1]
    
        # remove double two way edges
        edges = graph_funcs.remove_two_way_edges(edges)
    
        # remove non parallel opposite edges (or long way)
        edges = graph_funcs.remove_long_way_edges(edges)
    
        # allocated edge height based on cardinal method
        layer_allocation, _ = graph_funcs.allocate_edge_height(edges, 0)
    
        # Assign layer allocation to geodataframe
        edges['layer_height'] = layer_allocation
    
        # # add interior angles at all intersections
        edges = graph_funcs.add_edge_interior_angles(edges)
    
        # Perform COINS algorithm to add stroke groups
        self.edges = coins.COINS(edges)
        
        ##########################################################################
        # Until here, we have a graph, but the groups are undirected. We will apply
        # an initial condition.
    
        # set directionality of groups with one edge
        self.init_directions = [(33302019, 378727, 0),   # group 0
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
        
        
        # Start genetic algorithm
        creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
        creator.create("Individual", self.boollist, fitness = creator.FitnessMin)
        
        toolbox = base.Toolbox()
        
        toolbox.register("n_per_product", self.n_per_product)
        
        
        
    def boollist():
        return [random.randint(0,1) for i in range(45)]
        
    def n_per_product():
        return [0] * 45
    
    def CostFunction(self, bool_list):
        # reoroder edge geodatframe
        self.edges = graph_funcs.set_direction(self.edges, self.edge_directions)
        
        # create graph
        G = ox.graph_from_gdfs(self.nodes, self.edges)
        
        # Get cost
        cost = dijkstra_search_multiple(G, self.orig_nodes, self.dest_nodes)
        return cost
