# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 13:55:27 2021

@author: nipat
## code bassed on : https://www.redblobgames.com/pathfinding/a-star/implementation.html"""

import osmnx as ox
import matplotlib.pyplot as plt
import numpy as np
import heapq
from typing import Protocol, Dict, List, Iterator, Tuple, TypeVar, Optional
import os

#create a node class for each osmnx node
class Node:
    def __init__(self,key,x,y):
        self.key=key
        self.x=x
        self.y=y
        self.children={}# each element of neigh is like [ox_key,edge_cost] 
        

class PriorityQueue:
    def __init__(self):
        self.elements: List[Tuple[float, Node]] = []
    
    def empty(self) -> bool:
        return not self.elements
    
    def put(self, item: Node, priority: float):
        heapq.heappush(self.elements, (priority,item.key, item))
    
    def get(self) -> Node:
        return heapq.heappop(self.elements)[2]


def dijkstra_search_multiple(graph, starts, goals, printPaths = False):
    totalcost = 0
    for start in starts:
        costs = dijkstra_search(graph, start, goals, printPaths)
        totalcost += sum(costs.values())
    return totalcost


def dijkstra_search(graph, start, goals, printPaths=False):
    goals_duplicate=goals.copy()
    start.c=0#const to start from start is 0
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from: Dict[int, Optional[int]] = {}
    cost_so_far: Dict[int, float] = {}
    came_from[start.key] = None
    cost_so_far[start.key] = 0
    
    
    costs: Dict[int, int] = {}
    paths: Dict[int, [] ]= {}
    
    while not frontier.empty():
        current = frontier.get()

        if current.key in goal:
            #pathFound=True
            costs[current.key]=cost_so_far[current.key]
            goal.remove(current.key)
            if not goal:
                break
        
        for next in current.children.keys():
            new_cost = cost_so_far[current.key] +current.children[next]
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
                frontier.put(graph[next], priority)
                came_from[next] = current.key
         
    for g in goal:
        costs[g]=100000# if no path is foucn cost=-1
    if printPaths:
        for g in goals_duplicate:
            if not g in goal:
                route=[]
                tmp=(graph[g].x,graph[g].y)
                route.append(tmp)
                cc=came_from[g]
                while not cc==start.key:
                    tmp=(graph[cc].x,graph[cc].y)
                    route.append(tmp)
                    cc=came_from[cc]
                paths[g]=route
        return costs, paths
    return costs


##Load the street map
dir_path = os.path.dirname(os.path.realpath(__file__))
graph_path = dir_path.replace('graph_definition','graph_definition/gis/data/street_graph/processed_graph.graphml')
G = ox.io.load_graphml(graph_path)
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

##Define the start node
start_id=23
key=G_list[start_id]
start_node=graph[key] 
x_start=G._node[key]['x']
y_start=G._node[key]['y']

##Define the goals, it is a lost of osmnx keys 
goals=[]
# goal_id=6
# key=G_list[goal_id]
# goals.append(key)
# goal_id=80
# key=G_list[goal_id]
# goals.append(key)

for goal in [1,2,3,4,5,6,7,8,9,10, 11, 12,13,14,15,16,45,67,34,87,54,27]:
    goals.append(G_list[goal])

#here call the path planning algortihm, it needs defined a s a dictionary as shown above, 
##the start node and a list of the osmnx keys of the desired goal nodes
#the function returns a dictionary with the keys being the keys of the goal nodes 
##and the values of each key is the length of the path from the start to that node expressed in meters
##If you want the function to also return the paths you need to put True as the forth input, that will increase the memory and time consumption of the function
costs,paths= dijkstra_search(graph,start_node,goals.copy(),True)

print(start_node.key)
print(costs)

##Print paths
fig, ax = ox.plot_graph(G,node_color="w",show=False,close=False)
x_list=[]
y_list=[]
for g in paths.keys():
    for r in paths[g]:
        x_list.append(r[0])
        y_list.append(r[1])

ax.scatter(x_list,y_list, color='g')
ax.scatter(x_start,y_start, color='b')
plt.show()
