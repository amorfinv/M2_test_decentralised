# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 13:55:27 2021

@author: nipat
## code bassed on : https://www.redblobgames.com/pathfinding/a-star/implementation.html"""

import heapq
from typing import  Dict, List, Tuple, Optional
import copy

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
    return totalcost,


def dijkstra_search(graph, start, goal, printPaths=False):
    goals_duplicate=copy.copy(goal)
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