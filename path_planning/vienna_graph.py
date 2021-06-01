# -*- coding: utf-8 -*-
"""
Created on Wed May  5 12:19:13 2021

@author: nipat
"""
import osmnx
import matplotlib.pyplot as plt
import heapq

class Node:
    av_speed_horizontal=0.005#10.0
    av_speed_vertical=2.0
    def __init__(self,key_index,x,y,index):
        self.key_index=key_index # the index the osmnx graph
        self.index=index# the index in the search graph

        self.x=x
        self.y=y
        self.z=0

        #self.osmnx_parents=[]
        #self.osmnx_children=[]
        self.parents=[]
        self.children=[]
        
        self.f=0.0
        self.g=float('inf')
        self.rhs=float('inf')
        self.h=0.0
        self.key=[0.0,0.0]

        self.density=1.0 #shows the traffic density      

        self.inQueue=False
        self.speed=0.005
        
class Path:
    def __init__(self,start,goal):
        self.start=start
        self.goal=goal
        self.k_m=0
        self.queue=[]
        
def initialise(path):
    path.queue=[]
    path.k_m=0
    path.goal.rhs=0
   #heapq.heappush(path.queue, calculateKey(path.goal,path.start, path.k_m) + (path.goal,))
    path.goal.inQueue=True
    path.goal.h=heuristic(path.start,path.goal)
    heapq.heappush(path.queue, (path.goal.h,0,path.goal.x,path.goal.y,path.goal.z, path.goal))
   #print(path)   


def compare_keys(node1,node2):
    if node1[0]<node2[0]:
        return True
    elif node1[0]==node2[0] and node1[1]<node2[1]:
        return True
    return False
        
def calculateKey(node,start, k_m):
    return (min(node.g, node.rhs) + heuristic(node,start) + k_m, min(node.g, node.rhs))

def heuristic(current, goal):
    h=abs(goal.x-current.x)/current.av_speed_horizontal+abs(goal.y-current.y)/current.av_speed_horizontal+abs(goal.z-current.z)/current.av_speed_vertical
    return h

def compute_c(current,neigh):
    g=1
    if(current.z!=neigh.z):
        g=abs(neigh.z-current.z)/current.av_speed_vertical
    else:
        #g=(abs(neigh.x-current.x)+abs(neigh.y-current.y))*2/(current.speed/current.density+neigh.speed/neigh.density)
        g=G[current.key_index][neigh.key_index][0]['length']
    return g

def top_key(q):
    #q.sort()
    # print(queue)
    if len(q) > 0:
        return [q[0][0],q[0][1]]
    else:
        # print('empty queue!')
        return [float('inf'), float('inf')]
    
def update_vertex(path,node):
    if node.g!=node.rhs and node.inQueue:
        #Update
        node.key=calculateKey(node, path.start, path.k_m)
        id_in_queue = [item for item in path.queue if node in item]
        if id_in_queue != []:
            if len(id_in_queue) != 1:
                raise ValueError('more than one ' + node + ' in the queue!')
            path.queue.remove(id_in_queue[0])
            heapq.heappush(path.queue, (node.key[0],node.key[1],node.x,node.y,node.z,node))
    elif node.g!=node.rhs and not node.inQueue:
        #Insert
        node.inQueue=True
        node.h=heuristic(node, path.start)
        node.key=calculateKey(node, path.start, path.k_m)
        heapq.heappush(path.queue, (node.key[0],node.key[1],node.x,node.y,node.z,node))
    elif node.g==node.rhs and node.inQueue:
        #remove
        node.inQueue=False
        id_in_queue = [item for item in path.queue if id in item]
        if id_in_queue != []:
            if len(id_in_queue) != 1:
                raise ValueError('more than one ' + id + ' in the queue!')
            path.queue.remove(id_in_queue[0])
            
def compute_shortest_path(path):
    path.start.key=calculateKey(path.start, path.start, path.k_m)
    k_old=top_key(path.queue)
   
    while path.start.rhs>path.start.g or compare_keys(k_old,path.start.key):
        if len(path.queue)==0:
            print("No path found!")
            return 0
        #print(len(path.queue))
        k_old=top_key(path.queue)
        current_node=path.queue[0][5]#get the node with teh smallest priority
        k_new=calculateKey(current_node, path.start, path.k_m)
        if compare_keys(k_old, k_new):
            current_node.key=k_new
            id_in_queue = [item for item in path.queue if current_node is item]
            if id_in_queue != []:
                if len(id_in_queue) != 1:
                    raise ValueError('more than one ' + current_node + ' in the queue!')
                path.queue.remove(id_in_queue[0])
                heapq.heappush(path.queue, (current_node.key[0],current_node.key[1],current_node.x,current_node.y,current_node.z,current_node))
        elif current_node.g>current_node.rhs:
            current_node.g=current_node.rhs
            id_in_queue = [item for item in path.queue if current_node in item]
            if id_in_queue != []:
                if len(id_in_queue) != 1:
                    raise ValueError('more than one ' + current_node + ' in the queue!')
                path.queue.remove(id_in_queue[0])
                current_node.inQueue=False

                for p in current_node.parents:
                    pred_node=graph[p]  
                    if pred_node!=path.goal:
                        pred_node.rhs=min(pred_node.rhs,current_node.g+compute_c(pred_node,current_node))
                    update_vertex(path, pred_node)
        else:
            g_old=current_node.g
            current_node.g=float('inf')
            pred_node=current_node
            if pred_node.rhs==g_old:
                if pred_node!= path.goal:
                    #child_list=get_child_list(path, pred_node)
                    tt=()
                    for ch in pred_node.children:
                        child=graph[ch]
                        tt.append(child.g+compute_c(child, pred_node))
                    pred_node.rhs=min(tt)
            #parent_list=get_parent_list(path, current_node)
            for p in current_node.parents:
                parent=graph[p]
                if parent.rhs==(g_old+compute_c(current_node, parent)):
                    if(parent!=path.goal):
                    #child_list=get_child_list(path, pred_node)
                        tt=()
                        for ch in parent.children:
                            child=graph[ch]
                            tt.append(child.g+compute_c(child, parent))
                        parent.rhs=min(tt)
                update_vertex(path, parent)
            pred_node=current_node
            if pred_node.rhs==g_old:
                if pred_node!= path.goal:
                     #child_list=get_child_list(path, pred_node)
                    tt=()
                    for ch in pred_node.children:
                        child=graph[ch]
                        tt.append(child.g+compute_c(child, pred_node))
                    pred_node.rhs=min(tt)
            update_vertex(path, pred_node)
    return 1
            
            
def get_path(path):
    change=False
    change_list=[]# a list with the nodes between which the cost changed
    route=[]
    tmp=(path.start.x,path.start.y,path.start.z)
    route.append(tmp)
    while path.start!=path.goal:
        current_node=path.start
        #child=get_child_list(path, path.start)
        minim=float('inf')
        for ch in path.start.children:
            n=graph[ch]
            #print(compute_c(path.start, n)+n.g)
            if compute_c(path.start, n)+n.g<minim:
                minim=compute_c(path.start, n)+n.g
                current_node=n

        #find the intermediate points
        linestring=edges_geometry[path.start.key_index][current_node.key_index][0] #if teh start index should go first need to get checked
        coords = list(linestring.coords)
        for c in range(len(coords)-1):
            if not c==0:
                tmp=(coords[c][0],coords[c][1],current_node.z) #the intermediate point
                route.append(tmp) 
                
            
        tmp=(current_node.x,current_node.y,current_node.z) #the next node
        route.append(tmp) 
        if change: #Scan for changes
            path.k_m=path.k_m+heuristic(current_node, path.start)
            for c in change_list:
                c_old=compute_c(c[0], c[1])
                #update cost and obstacles here
                if c_old>compute_c(c[0], c[1]):
                    if(c[0]!=path.goal):
                        c[0].rhs=min(c[0].rhs,compute_c(c[0], c[1])+c[1].g)
                elif c[0].rhs== c_old+c[1].g:
                    if c[0]!=path.goal:
                        #child_list=get_child_list(path, c[0])
                        tt=[]
                        for ch in c[0].children:
                            child=graph[ch]
                            tt.append(child.g+compute_c(child, c[0]))
                        c[0].rhs=min(tt)
                update_vertex(path, c[0])
                compute_shortest_path(path)
                
        path.start=current_node
        
    return route

                        


osmnx.config(use_cache=True, log_console=True)
G = osmnx.graph_from_bbox(48.2037, 48.2129, 16.3636, 16.3781, network_type='drive')

#gdf_nodes = osmnx.graph_to_gdfs(G, edges=False, node_geometry=False)#[["x", "y"]]
edges_geometry=osmnx.graph_to_gdfs(G, nodes=False)["geometry"]

#G1 = osmnx.graph_from_bbox(48.21, 48.213, 16.3765, 16.378, network_type='drive')

print(G[685044][453544241][0]['length'])

#plt.scatter(48.213, 16.3765,color='g')
fig, ax = osmnx.plot_graph(G,node_color="w",show=False,close=False)

#ax.scatter(16.3765745,48.2122762, color='r')
#ax.scatter(16.3761943,48.2118166, color='b')
#ax.scatter(16.3766298,48.2123456, color='g')
#plt.show()
#fig, ax = osmnx.plot_graph(G1,node_color="b")
#G = osmnx.graph_from_bbox(48.117660, 48.322580, 6.577510, 16.18219, network_type='drive')


#start_key=685044
#goal_key=32665507
#x=G._node[start_key]['x']
#y=G._node[start_key]['y']
#start_node=Node(start_key,x,y)
#start_node.children=list(G._succ[start_key].keys())
#x=G._node[goal_key]['x']
#y=G._node[goal_key]['y']
#goal_node=Node(goal_key,x,y)
#goal_node.parents=list(G._pred[goal_key].keys())
#g_adj=G._adj[685044]
#g_degree=G.degree[685044]
#g_nodes=G.nodes[685044]


##Create the search graph
graph=[]
omsnx_keys_list=list(G._node.keys())
G_list=list(G._node)
if 1: #one layer
    for i in range(len(omsnx_keys_list)):
        key=omsnx_keys_list[i]
        x=G._node[key]['x']
        y=G._node[key]['y']
        node=Node(key,x,y,i)
        children=list(G._succ[key].keys())
        for ch in children: 
            node.children.append(G_list.index(ch))
        #node.children.append(i+1)
        parents=list(G._pred[key].keys())
        for p in parents:
            node.parents.append(G_list.index(p))
        graph.append(node)
if 0: #2 layers
    for i in range(len(omsnx_keys_list)):
        key=omsnx_keys_list[i]
        x=G._node[key]['x']
        y=G._node[key]['y']
        node=Node(key,x,y,2*i)
        children=list(G._succ[key].keys())
        for ch in children: 
            node.children.append(2*G_list.index(ch))
        node.children.append(2*i+1)
        parents=list(G._pred[key].keys())
        for p in parents:
            node.parents.append(2*G_list.index(p))
        node.parents.append(2*i+1)
        graph.append(node)
    
        node=Node(key,x,y,2*i+1)
        node.z=10
        for ch in children: 
            node.parents.append(2*G_list.index(ch)+1)
        node.parents.append(2*i)
        for p in parents:
            node.children.append(2*G_list.index(p)+1)
        node.children.append(2*i)
        graph.append(node)
if 0:# 3 layers
    for i in range(len(omsnx_keys_list)):
        key=omsnx_keys_list[i]
        x=G._node[key]['x']
        y=G._node[key]['y']
        node=Node(key,x,y,3*i)
        children=list(G._succ[key].keys())
        for ch in children: 
            node.children.append(3*G_list.index(ch))
        node.children.append(3*i+1)
        parents=list(G._pred[key].keys())
        for p in parents:
            node.parents.append(3*G_list.index(p))
        node.parents.append(3*i+1)
        graph.append(node)
    
        node=Node(key,x,y,3*i+1)
        node.z=10
        for ch in children: 
            node.parents.append(3*G_list.index(ch)+1)
        node.parents.append(3*i)
        for p in parents:
            node.children.append(3*G_list.index(p)+1)
        node.children.append(3*i)
        graph.append(node)
        
        node=Node(key,x,y,3*i+2)
        node.z=15
        for ch in children: 
            node.parents.append(3*G_list.index(ch)+2)
        node.parents.append(3*i)
        for p in parents:
            node.children.append(3*G_list.index(p)+2)
        node.children.append(3*i)
        graph.append(node)
        
        
        


start_id=160#14 #5,9,19,20,160
key=G_list[start_id]
#start_id=start_id*2 //only for 2 layers
start_node=graph[start_id] 
x_start=G._node[key]['x']
y_start=G._node[key]['y']

goal_id=0
goal_node=graph[goal_id] #0
#key=G_list[goal_id*2] //only fo rtwo layers
x_goal=G._node[key]['x']
y_goal=G._node[key]['y']

#print(G.edges._adjdict[685044][453544241][0]['name'])


path=Path(start_node,goal_node)

initialise(path)

path_found=compute_shortest_path(path)

route=[]
if path_found:
    route=get_path(path)
    
print(route)

x_list=[]
y_list=[]
x_list_up=[]
y_list_up=[]
for r in route:
    if(r[2]==0):
        x_list.append(r[0])
        y_list.append(r[1])
    else:
        x_list_up.append(r[0])
        y_list_up.append(r[1])
ax.scatter(x_list,y_list, color='g')
ax.scatter(x_list_up,y_list_up, color='y')
ax.scatter(x_start,y_start, color='b')
ax.scatter(x_goal,y_goal, color='r')
plt.show()
