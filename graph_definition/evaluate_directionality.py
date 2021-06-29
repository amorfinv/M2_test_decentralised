# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 13:55:27 2021

@author: nipat
## code bassed on : https://www.redblobgames.com/pathfinding/a-star/implementation.html"""

import osmnx
import matplotlib.pyplot as plt
import numpy as np
import heapq

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


def dijkstra_search(graph, start, goal,printPaths=False):
    goals_duplicate=goal.copy()
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
                
    if printPaths:
        for g in goals_duplicate:
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
G = osmnx.io.load_graphml(filepath='C:/Users/nipat/Downloads/M2_test_scenario-main/M2_test_scenario-main/graph_definition/gis/data/street_graph/processed_graph1.graphml')
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
goal_id=0
key=G_list[goal_id]
goals.append(key)
goal_id=56
key=G_list[goal_id]
goals.append(key)

#here call the path planning algortihm, it needs defined a s a dictionary as shown above, 
##the start node and a list of the osmnx keys of the desired goal nodes
#the function returns a dictionary with the keys being the keys of the goal nodes 
##and the values of each key is the length of the path from the start to that node expressed in meters
##If you want the function to also return the paths you need to put True as the forth input, that will increase the memory and time consumption of the function
costs,paths= dijkstra_search(graph,start_node,goals.copy(),True)

print(start_node.key)
print(costs)

##Print paths
fig, ax = osmnx.plot_graph(G,node_color="w",show=False,close=False)
x_list=[]
y_list=[]
for g in goals:
    for r in paths[g]:
        x_list.append(r[0])
        y_list.append(r[1])

ax.scatter(x_list,y_list, color='g')
ax.scatter(x_start,y_start, color='b')
plt.show()
