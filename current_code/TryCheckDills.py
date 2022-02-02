# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:52:01 2022

@author: nipat
"""

plan=1
plan_name=""


        self.in_same_cell=False
        self.init_succesful=True

        self.path_only_open=False
        ###
 
try:
    a=plan.init_succesful
    if not plan.init_succesful:
        print(plan_name,"init_succesful not succ") 
except:
    print(plan_name,"init_succesful")  
try:
    a=plan.in_same_cell
except:
    print(plan_name,"in_same_cell")   

if  not plan.in_same_cell:
    try:
        a=plan.os_keys2_indices
    except:
        print(plan_name,"os_keys2_indices")   
    try:
        a=plan.graph
    except:
        print(plan_name,"graph")   
     try:
        a=plan.graph.start_point
    except:
        print(plan_name,"graph.start_point")   
    try:
        a=plan.graph.goal_point
    except:
        print(plan_name,"graph.goal_point")   
    try:
        a=plan.graph.start_ind
    except:
        print(plan_name,"graph.start_ind")           
    try:
        a=plan.graph.key_indices_list
    except:
        print(plan_name,".graph.key_indices_list")    
    try:
        a=plan.graph.groups_list
    except:
        print(plan_name,".graph.groups_list")            
    try:
        a=plan.graph.g_list
    except:
        print(plan_name,".graph.g_list")    
    try:
        a=plan.graph.rhs_list
    except:
        print(plan_name,".graph.rhs_list")   
    try:
        a=plan.graph.key_list
    except:
        print(plan_name,".graph.key_list")   
    try:
        a=plan.graph.inQueue_list
    except:
        print(plan_name,".graph.inQueue_list")           
    try:
        a=plan.graph.expanded_list
    except:
        print(plan_name,".graph.expanded_list")   
    try:
        a=plan.graph.parents_list
    except:
        print(plan_name,".graph.parents_list")   
    try:
        a=plan.graph.children_list 
    except:
        print(plan_name,".graph.children_list ")           
    try:
        a=plan.path.start
    except:
        print(plan_name,"path.start ")  
    try:
        a=plan.path.goal
    except:
        print(plan_name,"path.goal ")          
    try:
        a=plan.path.k_m
    except:
        print(plan_name,"path.k_m")   
    try:
        a=plan.path.queue
    except:
        print(plan_name,"path.queue")  
    try:
        a=plan.path.speed
    except:
        print(plan_name,"path.speed)          
     try:
        a=plan.path.graph
    except:
        print(plan_name,"path.graph")              
 

try:
    a=plan.path_only_open
except:
    print(plan_name,"path_only_open")  
try:
    a=plan.edges_list
except:
    print(plan_name,"edges_list")      
try:
    a=plan.next_turn_point
except:
    print(plan_name,"next_turn_point")  
try:
    a=plan.groups
except:
    print(plan_name,"groups")      
try:
    a=plan.turn_speed
except:
    print(plan_name,"turn_speed")      
try:
    a=plan.in_constrained
except:
    print(plan_name,"in_constrained")  
     
try:
    a=plan.speed_max
except:
    print(plan_name,"speed_max")
try:
    a=plan.route
except:
    print(plan_name,"route")
try:
    a=plan.turns
except:
    print(plan_name,"turns")
try:
    a=plan.priority
except:
    print(plan_name,"priority")    
try:
    a=plan.loitering
except:
    print(plan_name,"loitering")    
try:
    a=plan.loitering_edges
except:
    print(plan_name,"loitering_edges")    
try:
    a=plan.start_point
except:
    print(plan_name,"start_point")   

try:
    a=plan.goal_point
except:
    print(plan_name,"goal_point")   
try:
    a=plan.cutoff_angle
except:
    print(plan_name,"cutoff_angle")       
    
try:
    a=plan.aircraft_type
except:
    print(plan_name,"aircraft_type")
try:
    a=plan.start_index
except:
    print(plan_name,"start_index")
try:
    a=plan.start_index_previous
except:
    print(plan_name,"start_index_previous")
try:
    a=plan.start_in_open
except:
    print(plan_name,"start_in_open")
try:
    a=plan.goal_index
except:
    print(plan_name,"goal_index")
try:
    a=plan.goal_index_next
except:
    print(plan_name,"goal_index_next")
try:
    a=plan.dest_in_open
except:
    print(plan_name,"dest_in_open")
try:
    a=plan.path
except:
    print(plan_name,"path")
##do the path vari