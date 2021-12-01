# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 11:22:00 2021

@author: nipat
"""

import math
import copy
#import osmnx as ox
from pyproj import Proj, transform

# Given three collinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def onSegment(p, q, r):
    if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
            (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
        return True
    return False  
                
def ccw(A,B,C):
    if (C[1]-A[1]) * (B[0]-A[0]) == (B[1]-A[1]) * (C[0]-A[0]):
        return 2
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    if ccw(A,B,C)==2 and onSegment(A,C,B):
        return True
    elif ccw(A,B,D)==2 and onSegment(A,D,B):
        return True
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def line_intersection_point(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y
  


                
class Cell:
    def __init__(self):
        self.id=None
        self.key_index=None
        self.p0=None#[x,y]
        self.p1=None
        self.p2=None
        self.p3=None
 
        self.center_x=None
        self.center_y=None
        
        self.neighbors=[]
        
        self.constrained_geometry=[]
        
        self.constrained_adj_up=False
        self.constrained_adj_down=False
        self.constrained_adj_half_up=False
        self.constrained_adj_half_down=False
        self.y_lower=None
        self.y_max=None
        
        self.entry_list=[]
        self.exit_list=[]
                
        self.minimum_altitude=0
        

         
class open_airspace:
    line_extension=1 # TODO : check if that value is fine  
    def __init__(self,geovectors,poly_points):
        self.geovectors=geovectors #list of geovectors with [[p1_x,p1,y],...]
        self.poly_points=poly_points
        self.geo_dict={}
        self.coords_x_list=[]
        self.grid=[]
        self.world_center_x=601000 # TODO : find the exact center coordinates
        self.world_center_y=5340000
        self.world_radius=9000#8000 #in meters
        self.constrained_center=[]
        
        self.entry_cells_dict={}
        self.exit_cells_dict={}
        self.points_border_dict={} # keys [x][y], value true if it is on the constrained border
        
        
        
       # self.geovect_points_inconstrained=[]#[self.geovectors[15][1][0],self.geovectors[5][3][0],self.geovectors[12][0][0],self.geovectors[12][3][0]]#[[15,2],[5,3],[12,0],[12,3]] #[number of geovector, number of point]

        
        self.create_geovector_dict()
        self.create_grid_lines()
        self.create_grid()
        self.manual_grid_modifications()
        
        self.entry_points_list=[]
        self.exit_points_list=[]
        self.entry_nodes_dict={}
        self.exit_nodes_dict={}
        

        
        self.connect_opengraph2constrained()
        
        self.key_index_2_graph_dict={}# TODO : fill in this dictionary
        self.neighboring_edges_dict={}# TODO : fill in this dictionary
        self.fill_dictionaries()


    def fill_dictionaries(self):
        
        for i,cell in enumerate(self.grid):
            self.key_index_2_graph_dict[cell.key_index]=i

        for i,cell in enumerate(self.grid):
            tmp={}
            for n in cell.neighbors:
                cc=self.grid[self.key_index_2_graph_dict[n]]
                px1=1
                py1=1
                px2=2
                py2=2
                tmp[n]=[[px1,py1],[px2,py2]]
            self.neighboring_edges_dict[cell.key_index]=tmp
                
            
    def manual_grid_modifications(self):

        
        #Shrink cell # 35
        cell=self.grid[35]
        p0y=cell.p2[1]
        p1y=cell.p2[1]
        cell.p2[1]=cell.p1[1]#
        
        cell.constrained_adj_down=False
        cell.center_x=(cell.p0[0]+cell.p1[0]+cell.p2[0]+cell.p3[0])/4
        cell.center_y=(cell.p0[1]+cell.p1[1]+cell.p2[1]+cell.p3[1])/4
        x=cell.center_x
        y=cell.center_y
# =============================================================================
#         #init_key_index=cell.key_index
#         #cell.key_index=int((-(x+y)//1+(x-(x//1))*10000000000+(y-(y//1))*10000000000)//1)
#         
#         for nn in cell.neighbors:
#             self.grid
#             n.neighbors.remove(init_key_index)
#             n.neighbors.appen(cell.key_index)
# =============================================================================
        
        #Add cell down to # 35
        c=Cell()
        c.id=[None,cell.p2[0]]
        c.p0=[cell.p1[0],cell.p1[1]]
        c.p1=[cell.p1[0],p1y]
        c.p2=[cell.p2[0],p1y]
        c.p3=[cell.p2[0],cell.p1[1]]
        c.constrained_geometry=cell.constrained_geometry
        c.y_lower=cell.y_lower
        c.constrained_adj_down=True
        c.center_x=(c.p0[0]+c.p1[0]+c.p2[0]+c.p3[0])/4
        c.center_y=(c.p0[1]+c.p1[1]+c.p2[1]+c.p3[1])/4
        x=c.center_x
        y=c.center_y
        c.key_index=int((-(x+y)//1+(x-(x//1))*10000000000+(y-(y//1))*10000000000)//1)
        c.neighbors.append(cell.key_index)
        c.neighbors.append(self.grid[32].key_index)
        self.grid.append(c)
        
        self.grid[32].neighbors.append(c.key_index)
        
        cell.neighbors.append(c.key_index)
        cell.constrained_geometry=[]
        cell.y_lower=None


        
        #expand cell # 70
        cell=self.grid[70]
        p0y=cell.p2[1]
        cell.p2[1]=cell.constrained_geometry[len(cell.constrained_geometry)-11][1] 
        p1y=cell.p2[1]
        p1y_old=cell.p1[1]
        cell.p1[1]=cell.constrained_geometry[38][1]
        
        #Add cell right to # 70
        c=Cell()
        c.id=[None,cell.p2[0]]
        c.p0=[cell.p0[0]+50,p1y_old] # TODO : maybe that is wrong and the +100 shouldn't be here but as a special case in the connect_opengraph2constrained() function
        c.p1=[cell.p0[0]+50,cell.p1[1]]
        c.p2=[cell.p0[0],cell.p1[1]]
        c.p3=[cell.p0[0],p1y_old]
        c.constrained_geometry=[]
        c.y_lower=cell.y_lower
        c.constrained_adj_down=True
        c.center_x=(c.p0[0]+c.p2[0])/2
        c.center_y=(c.p0[1]+c.p1[1])/2
        x=c.center_x
        y=c.center_y
        c.key_index=int((-(x+y)//1+(x-(x//1))*10000000000+(y-(y//1))*10000000000)//1)
        c.neighbors.append(cell.key_index)
        
        for i in range(38):
            c.constrained_geometry.append(cell.constrained_geometry[0])
            del cell.constrained_geometry[0]

        self.grid.append(c)
        
        cell.neighbors.append(c.key_index)

        
        #Add cell left to # 70
        c=Cell()
        c.id=[None,cell.p2[0]]
        c.p0=[cell.p2[0],p0y]
        c.p1=[cell.p2[0],p1y]
        x2=self.world_center_x-self.world_radius
        x23=598767.6635372492
        y23=5343053.60999253
        c.p2=[x23,y23]
        c.p3=[x23,y23]
        c.constrained_geometry=[]
        for i in range(len(cell.constrained_geometry)-12,len(cell.constrained_geometry)):
            c.constrained_geometry.append(cell.constrained_geometry[i])
        for i in range(len(cell.constrained_geometry)-12,len(cell.constrained_geometry)):
            del cell.constrained_geometry[-1]
            
        c.y_lower=cell.y_lower
        c.constrained_adj_down=True
        c.center_x=(c.p0[0]+c.p2[0])/2
        c.center_y=(c.p0[1]+c.p1[1]+c.p2[1])/3
        x=c.center_x
        y=c.center_y
        c.key_index=int((-(x+y)//1+(x-(x//1))*10000000000+(y-(y//1))*10000000000)//1)
        c.neighbors.append(cell.key_index)
        self.grid.append(c)
        
        cell.neighbors.append(c.key_index)
        



         #expand cell # 58
         ## TODO : constrained geometry_here is not estimated for cell #58 or for the added
        cell=self.grid[58] 
        cell.constrained_adj_down=True
        p1=cell.p1
        p2=copy.deepcopy(cell.p1)
        p2[1]=p2[1]-1000
        px=0
        py=0
        intersections=0
        for i in range(len(self.poly_points[len(self.poly_points)-1])):
            pp=self.poly_points[len(self.poly_points)-1][i]
            pp_next=self.poly_points[len(self.poly_points)-1][(i+1)%len(self.poly_points[len(self.poly_points)-1])]
            boolean=intersect(p1,p2,pp,pp_next)
            if boolean:
                intersections=intersections+1
            if intersections==3:
                px,py=line_intersection_point([p1,p2],[pp,pp_next])

                break
        
        cell.p1[1]=py
        cell.y_lower=cell.p1[1]

        #Add cell right to # 58 ## cell #222
        c=Cell()
        c.id=[None,cell.p0[0]]
        px1=598497.9147875205
        py1= 5343367.390299891
        c.p0=[px1,py1]
        c.p1=[px1,py1]
        c.p2=[cell.p0[0],py]
        c.p3=[cell.p0[0],cell.p1[1]]
        #c.constrained_geometry=cell.constrained_geometry
        c.y_lower=py1
        c.constrained_adj_down=True
        c.center_x=(c.p0[0]+c.p2[0])/2
        c.center_y=(c.p0[1]+c.p3[1]+c.p2[1])/3
        x=c.center_x
        y=c.center_y
        c.key_index=int((-(x+y)//1+(x-(x//1))*10000000000+(y-(y//1))*10000000000)//1)
        c.neighbors.append(cell.key_index)
        self.grid.append(c)
        
        cell.neighbors.append(c.key_index)
        
        
        #expand cell # 72
        cell=self.grid[72] 
        cell.constrained_adj_up=True
        p1=cell.p3
        p2=copy.deepcopy(cell.p3)
        p2[1]=p2[1]+1000
        
        p3=[599145.7829077174,5336619.415107749]
        px=0
        py=0
        intersection_index1=-1
        intersection_index2=-1
        intersections=0
        intersections2=0
        for i in range(len(self.poly_points[len(self.poly_points)-1])):
            pp=self.poly_points[len(self.poly_points)-1][i]
            pp_next=self.poly_points[len(self.poly_points)-1][(i+1)%len(self.poly_points[len(self.poly_points)-1])]
            boolean=intersect(p1,p2,pp,pp_next)
            boolean2=intersect(p1,p3,pp,pp_next)
            if boolean:
                intersections=intersections+1
            if boolean2:
                intersections2=intersections2+1
            if intersections==3:
                px,py=line_intersection_point([p1,p2],[pp,pp_next])
                intersection_index2=i
                break
            if intersections2==3:
                px,py=line_intersection_point([p1,p2],[pp,pp_next])
                intersection_index1=i+1

        
        p3y_old=cell.p3[1]
        cell.p3[1]=py
        cell.y_upper=cell.p3[1]

        #Add cell right to # 72 ## cell #223
        c=Cell()
        c.id=[None,cell.p3[0]]
        px1=599145.7829077174
        py1=5336619.415107749
        c.p0=[cell.p3[0],cell.p3[1]]
        c.p1=[cell.p3[0],p3y_old]
        c.p2=[px1,py1]
        c.p3=[px1,py1]

        c.y_max=c.p0[1]
        c.constrained_adj_up=True
        c.center_x=(c.p0[0]+c.p2[0])/2
        c.center_y=(c.p0[1]+c.p1[1]+c.p2[1])/3
        x=c.center_x
        y=c.center_y
        c.key_index=int((-(x+y)//1+(x-(x//1))*10000000000+(y-(y//1))*10000000000)//1)
        c.neighbors.append(cell.key_index)
        for i in range(intersection_index1,intersection_index2+1):
            c.constrained_geometry.append(self.poly_points[len(self.poly_points)-1][i])
            if self.poly_points[len(self.poly_points)-1][i][1]>c.y_max:
                c.y_max=self.poly_points[len(self.poly_points)-1][i][1]
        
        self.grid.append(c)
        
        cell.neighbors.append(c.key_index)




        
    def create_geovector_dict(self):
        for i in range(len(self.geovectors)):
            for p in self.geovectors[i]:
                self.geo_dict[p[0]]=[p[1],i]
                self.coords_x_list.append(p[0])
                
                if i ==len(self.geovectors)-1:
                    self.points_border_dict[str(p[0])+"-"+str(p[1])]=True
                else:
                    self.points_border_dict[str(p[0])+"-"+str(p[1])]=False
                    

        
# =============================================================================
#     def connect_opengraph2constrained(self):
#      
#          G = ox.io.load_graphml('FINAL_GRAPH.graphml')
#         
#          f = open('entries.txt','r')      
#          contents = f.read()
#          f.close()
#          self.entry_points_list=[int(x) for x in contents.split(",")]
#          
#          f = open('exits.txt','r')
#          contents = f.read()
#          f.close()
#          self.exit_points_list=[int(x) for x in contents.split(",")]    
#          
#          for index in self.entry_points_list:
#              lon=G._node[index]['x']
#              lat=G._node[index]['y']
#              outProj = Proj(init='epsg:32633')
#              inProj = Proj(init='epsg:4326')
#              x,y = transform(inProj,outProj,lon,lat)
#              self.entry_nodes_dict[index]=[x,y]
#              
#          for index in self.exit_points_list:
#              lon=G._node[index]['x']
#              lat=G._node[index]['y']
#              inProj = Proj(init='epsg:4326')
#              outProj = Proj(init='epsg:32633')
#              x,y = transform(inProj,outProj,lon,lat)
#              self.exit_nodes_dict[index]=[x,y]
#          
#             
#          #lists to check for duplicates
#          entry_debug_list=[]
#          exit_debug_list=[]
#          
#          entry_cells_dict={}
#          exit_cells_dict={}
#          for index in self.entry_points_list:
#              entry_cells_dict[index]=[]
#          for index in self.exit_points_list:
#              exit_cells_dict[index]=[]
#          
#          for ii,cell in enumerate(self.grid):
#              
# 
#              if ii==219:
#                  for index in self.entry_points_list:
#                     [x,y]=self.entry_nodes_dict[index]
#                     if cell.p2[0]<x<cell.p0[0]+150:
#                         if cell.p0[1]>y>cell.p2[1]:
#                             cell.entry_list.append(index)
#                             entry_debug_list.append(index)
#                             entry_cells_dict[index].append(ii)
#                  for index in self.exit_points_list:
#                     [x,y]=self.exit_nodes_dict[index]
#                     if cell.p2[0]<x<cell.p0[0]+150:
#                         if cell.p0[1]>y>cell.p2[1]:
#                             cell.exit_list.append(index)
#                             exit_debug_list.append(index)
#                             exit_cells_dict[index].append(ii)
#                  continue
#              
#              if ii==46:
#                  for index in self.entry_points_list:
#                     [x,y]=self.entry_nodes_dict[index]
#                     if cell.p2[0]<x<cell.p0[0]+50:
#                         if cell.p0[1]>y>cell.p2[1]:
#                             cell.entry_list.append(index)
#                             entry_debug_list.append(index)
#                             entry_cells_dict[index].append(ii)
#                  for index in self.exit_points_list:
#                     [x,y]=self.exit_nodes_dict[index]
#                     if cell.p2[0]<x<cell.p0[0]+50:
#                         if cell.p0[1]>y>cell.p2[1]:
#                             cell.exit_list.append(index)
#                             exit_debug_list.append(index)
#                             exit_cells_dict[index].append(ii)
#              if ii==195:
#                  for index in self.entry_points_list:
#                     [x,y]=self.entry_nodes_dict[index]
#                     if cell.p2[0]-10<x<cell.p0[0]:
#                         if cell.p0[1]>y>cell.p3[1]:
#                             cell.entry_list.append(index)
#                             entry_debug_list.append(index)
#                             entry_cells_dict[index].append(ii)
#                  for index in self.exit_points_list:
#                     [x,y]=self.exit_nodes_dict[index]
#                     if cell.p2[0]-10<x<cell.p0[0]:
#                         if cell.p0[1]>y>cell.p3[1]:
#                             cell.exit_list.append(index)
#                             exit_debug_list.append(index)
#                             exit_cells_dict[index].append(ii)     
#              if ii==222:
#                  for index in self.entry_points_list:
#                     [x,y]=self.entry_nodes_dict[index]
#                     if cell.p2[0]<x<cell.p0[0]:
#                         if cell.p3[1]>y>cell.p1[1]:
#                             cell.entry_list.append(index)
#                             entry_debug_list.append(index)
#                             entry_cells_dict[index].append(ii)
#                  for index in self.exit_points_list:
#                     [x,y]=self.exit_nodes_dict[index]
#                     if cell.p2[0]<x<cell.p0[0]:
#                         if cell.p3[1]>y>cell.p1[1]:
#                             cell.exit_list.append(index)
#                             exit_debug_list.append(index)
#                             exit_cells_dict[index].append(ii) 
#                  continue
# 
# 
#                  
#              if cell.constrained_adj_up :
#                  for index in self.entry_points_list:
#                     [x,y]=self.entry_nodes_dict[index]
#                     if cell.p2[0]<=x<=cell.p0[0]:
#                         if cell.p1[1]<=y<=cell.y_max:#+1000:
#                             cell.entry_list.append(index)   
#                             entry_debug_list.append(index)
#                             entry_cells_dict[index].append(ii)
#                  for index in self.exit_points_list:
#                     [x,y]=self.exit_nodes_dict[index]
#                     if cell.p2[0]<=x<=cell.p0[0]:
#                         if cell.p1[1]<=y<=cell.y_max:#+1000:
#                             cell.exit_list.append(index)
#                             exit_debug_list.append(index)
#                             exit_cells_dict[index].append(ii)
#              elif cell.constrained_adj_down:
#                  for index in self.entry_points_list:
#                     [x,y]=self.entry_nodes_dict[index]
#                     if cell.p2[0]<=x<=cell.p1[0]:
#                         if cell.p0[1]>=y>=cell.y_lower:#-1000:
#                             cell.entry_list.append(index)
#                             entry_debug_list.append(index)
#                             entry_cells_dict[index].append(ii)
#                  for index in self.exit_points_list:
#                     [x,y]=self.exit_nodes_dict[index]
#                     if cell.p2[0]<=x<=cell.p1[0]:
#                         if cell.p0[1]>=y>=cell.y_lower:#-1000:
#                             cell.exit_list.append(index)
#                             exit_debug_list.append(index)
#                             exit_cells_dict[index].append(ii)
#                             
# 
# # =============================================================================
# ### This is probably not needed because there are nodes in teh edges of cells
# #
# #          ####Delete duplicates
# #          for index in self.entry_points_list:
# #              if len(entry_cells_dict[index])>1:
# #                  if len(entry_cells_dict[index])>2:
# #                      print(str(index)+":in more than 2 cells")
# #                  cell1=self.grid[entry_cells_dict[index][0]]
# #                  cell2=self.grid[entry_cells_dict[index][1]]
# #                  entry_debug_list.remove(index)
# #                  [x,y]=self.entry_nodes_dict[index]
# #                  in_cell1=False
# #                  in_cell2=False
# #                  
# #                  if cell1.constrained_adj_up :
# #                     if cell1.p2[0]<x<cell1.p0[0]:
# #                         if min(cell1.p1[1],cell1.p2[1])<y<cell1.y_max:
# #                             in_cell1=True
# #                  elif cell1.constrained_adj_down:
# #                     if cell1.p2[0]<x<cell1.p0[0]:
# #                         if cell1.p0[1]>y>cell1.y_lower:
# #                             in_cell1=True
# #                  if cell2.constrained_adj_up :
# #                     if cell2.p2[0]<x<cell2.p0[0]:
# #                         if min(cell2.p1[1],cell2.p2[1])<y<cell2.y_max:
# #                             in_cell2=True
# #                  elif cell2.constrained_adj_down:
# #                     if cell2.p2[0]<x<cell2.p0[0]:
# #                         if cell2.p0[1]>y>cell2.y_lower:
# #                             in_cell2=True
# #                             
# #                  if in_cell1 and in_cell2:
# #                     print(str(index)+":error: node in two cells")
# #                  elif in_cell1:
# #                     cell2.entry_list.remove(index)
# #                  elif in_cell2:
# #                     cell1.entry_list.remove(index)
# #                  else:
# #                     print(str(index)+":error in no cell")
# #                     
# #          for index in self.exit_points_list:
# #              if len(exit_cells_dict[index])>1:
# #                  if len(exit_cells_dict[index])>2:
# #                      print(str(index)+":in more than 2 cells")
# #                  cell1=self.grid[exit_cells_dict[index][0]]
# #                  cell2=self.grid[exit_cells_dict[index][1]]
# #                  exit_debug_list.remove(index)
# #                  [x,y]=self.exit_nodes_dict[index]
# #                  in_cell1=False
# #                  in_cell2=False
# #                  
# #                  if cell1.constrained_adj_up :
# #                     if cell1.p2[0]<x<cell1.p0[0]:
# #                         if min(cell1.p1[1],cell1.p2[1])<y<cell1.y_max:
# #                             in_cell1=True
# #                  elif cell1.constrained_adj_down:
# #                     if cell1.p2[0]<x<cell1.p0[0]:
# #                         if cell1.p0[1]>y>cell1.y_lower:
# #                             in_cell1=True
# #                  if cell2.constrained_adj_up :
# #                     if cell2.p2[0]<x<cell2.p0[0]:
# #                         if min(cell2.p1[1],cell2.p2[1])<y<cell2.y_max:
# #                             in_cell2=True
# #                  elif cell2.constrained_adj_down:
# #                     if cell2.p2[0]<x<cell2.p0[0]:
# #                         if cell2.p0[1]>y>cell2.y_lower:
# #                             in_cell2=True
# #                             
# #                  if in_cell1 and in_cell2:
# #                     print(str(index)+":error: node in two cells")
# #                  elif in_cell1:
# #                     cell2.exit_list.remove(index)
# #                  elif in_cell2:
# #                     cell1.exit_list.remove(index)
# #                  else:
# #                     print(str(index)+":error in no cell")
# # 
# #          
# #          if len(entry_debug_list)==len(set(entry_debug_list)):
# #             print("no entry duplicates")
# #          elif len(entry_debug_list)>=len(set(entry_debug_list)):
# #             print("entry duplicates!!!")
# #          if len(exit_debug_list)==len(set(exit_debug_list)):
# #             print("no exit duplicates")
# #          elif len(exit_debug_list)>=len(set(exit_debug_list)):
# #             print("exit duplicates!!!")
# # =============================================================================
#             
#          self.entry_cells_dict=entry_cells_dict
#          self.exit_cells_dict=exit_cells_dict
# =============================================================================
         
    def create_grid_lines(self):    
        ##Assuming no two vertices have the same x
        
        ##Do trapezoidal decomposition
        #https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng786/hw3.html
        self.coords_x_list.sort() #ascending
        self.grid=[]
        self.next_grids={}
        

        
###Check if a x value exists twice
        for i in range(len(self.coords_x_list)):
            cnt=0
            for j in range(len(self.coords_x_list)):
                if self.coords_x_list[i]==self.coords_x_list[j]:
                    cnt=cnt+1
            if cnt>1:
                print(self.coords_x_list[i],cnt)


        ##Create the lines
        lines={}#[[x, y_up, y_down]] if y_up or y_down=None no semiline
        for i in range(len(self.coords_x_list)):
            x=self.coords_x_list[i]

            
            y=self.geo_dict[x][0]
            
            d=self.world_radius*self.world_radius-(x-self.world_center_x)*(x-self.world_center_x)
            sqr=math.sqrt(d)
            y1=sqr+self.world_center_y
            y2=-sqr+self.world_center_y
            p1=[x,y1]
            p2=[x,y2]
            p3=[x,y]

            intersect_up=-1
            intersect_down=-1
            intersect_up_constrained=False
            intersect_down_constrained=False
            semiline_up=True
            semiline_down=True
            for geo_i in range(len(self.poly_points)):
                g=self.poly_points[geo_i]
                for j in range(len(g)):
                    pp=g[j]
                    pp_next=g[(j+1)%(len(g))]

                    if pp[0]!=x and pp_next[0]!=x:
                        boolean=intersect(p1,p3,pp,pp_next)
                        if boolean:
                            px,py=line_intersection_point([p1,p3],[pp,pp_next])
                            if intersect_up<0:
                                intersect_up=py
                                if geo_i==(len(self.poly_points)-1):
                                    intersect_up_constrained=True
                                else:
                                    intersect_up_constrained=False
                            elif intersect_up>py:
                                intersect_up=py
                                if geo_i==(len(self.poly_points)-1):
                                    intersect_up_constrained=True
                                else:
                                    intersect_up_constrained=False
                                    
                            if self.geo_dict[x][1]==geo_i and intersect_up==py:
                                semiline_up=False
                            elif intersect_up==py:
                                semiline_up=True
                              
                                
                        boolean=intersect(p3,p2,pp,pp_next)
                        if boolean:
                            px,py=line_intersection_point([p3,p2],[pp,pp_next])
                            if intersect_down<0:
                                intersect_down=py
                                if geo_i==(len(self.poly_points)-1):
                                    intersect_down_constrained=True
                                else:
                                    intersect_down_constrained=False
                            elif intersect_down<py:
                                intersect_down=py
                                if geo_i==(len(self.poly_points)-1):
                                    intersect_down_constrained=True
                                else:
                                    intersect_down_constrained=False
                                    
  
                            if self.geo_dict[x][1]==geo_i and intersect_down==py:
                                semiline_down=False
                            elif intersect_down==py:
                                semiline_down=True
                                
            
             
            if intersect_up==-1:
                y_up=y1
            else:
                y_up=intersect_up+self.line_extension ## Extend lines just a bit so that intersections are detected, 
            if not semiline_up:
                y_up=None
                
            if intersect_down==-1:
                y_down=y2
            else:
                y_down=intersect_down-self.line_extension
            if not semiline_down:
                y_down=None

            
            lines[x]=[x,y_up,y_down]

            if y_up is not None:
                if intersect_up_constrained:
                    self.points_border_dict[str(x)+"-"+str(y_up-self.line_extension)]=True
                else:
                    self.points_border_dict[str(x)+"-"+str(y_up-self.line_extension)]=False
            if y_down is not None:
                if intersect_down_constrained:
                    self.points_border_dict[str(x)+"-"+str(y_down+self.line_extension)]=True
                else:
                    self.points_border_dict[str(x)+"-"+str(y_down+self.line_extension)]=False
                    
        self.lines=lines    
        
    def create_grid(self):
        print("Create grid")
        
         ##Create the grid  
        print("Find next line")
        self.next_x=[]
        for i in range(len(self.coords_x_list)):
            print(i)
            x=self.coords_x_list[i]

            y=self.geo_dict[x][0]

            tmp=[]
            for j in range(i+1,len(self.coords_x_list)):
                x1=self.coords_x_list[j]
                
                y1=self.geo_dict[x1][0]
                p1=[x,y]
                p2=[x1,y1]
                
                intersects=False
                for l in self.lines.values():
                    if l[0] != x and l[0] !=x1:
                        p3=[l[0],self.geo_dict[l[0]][0]]
                        if l[1] is not None:
                            p4=[l[0],l[1]]
                            boolean=intersect(p1,p2,p3,p4)
                            if boolean:
                                intersects=True
                                break
                        if l[2] is not None:
                            p4=[l[0],l[2]]
                            boolean=intersect(p1,p2,p3,p4)
                            if boolean:
                                intersects=True
                                break
                            
                if intersects:
                    continue
                
                for g in self.poly_points:
                    for jj in range(len(g)):
                        p3=[g[jj][0],g[jj][1]]
                        p4=[g[(jj+1)%len(g)][0],g[(jj+1)%len(g)][1]]
                        if p3[0]!=x  and p4[0]!=x and  p3[0]!=x1  and p4[0]!=x1:
                            boolean=intersect(p1,p2,p3,p4)
                            if boolean:
                                intersects=True
                                break

                        p4=[g[(jj+2)%len(g)][0],g[(jj+2)%len(g)][1]]
                        if p3[0]!=x  and p4[0]!=x and  p3[0]!=x1  and p4[0]!=x1:
                            boolean=intersect(p1,p2,p3,p4)
                            if boolean:
                                intersects=True
                                break

                if not intersects:
                    tmp.append(x1)
                     
            next_grids=tmp
            self.next_grids[x]=next_grids
            
            self.next_x.append([x,tmp])

            
        print("Create cells")    
        for i in range(len(self.coords_x_list)):
            x=self.coords_x_list[i]
            y=self.geo_dict[x][0]
            next_grids=self.next_grids[x]
            
            if i==0:##Only for the first
                c=Cell()
                c.id=[None,x]
                c.p0=[x,self.lines[x][1]]
                c.p1=[x,self.lines[x][2]]
                x2=self.world_center_x-self.world_radius
                c.p2=[x2,self.lines[x][2]]
                c.p3=[x2,self.lines[x][1]]
                self.grid.append(c)
                
            elif i==len(self.coords_x_list)-1: ##only for the last one
                c=Cell()
                c.id=[x,None]
                c.p3=[x,self.lines[x][1]]
                c.p2=[x,self.lines[x][2]]
                x2=self.world_center_x+self.world_radius
                c.p1=[x2,self.lines[x][2]]
                c.p0=[x2,self.lines[x][1]]
                self.grid.append(c)    
            
            for j in range(len(next_grids)):
                c=Cell()
                c.id=[x,next_grids[j]]
                x_n=next_grids[j]
                y_n_up=self.lines[x_n][1]
                
                y_n_down=self.lines[x_n][2]
                y_n=self.geo_dict[x_n][0]
                
                y_up=self.lines[x][1]
                y_down=self.lines[x][2]
                
                if y_up is None and y_down is None:
                    continue
                if y_n_up is None and y_n_down is None:
                    continue

                #Undo the line extension
                if y_n_up is not None:
                    y_n_up=y_n_up-self.line_extension
                if y_n_down is not None:
                    y_n_down=y_n_down+self.line_extension
                if y_up is not None:
                    y_up=y_up-self.line_extension
                if y_down is not None:
                    y_down=y_down+self.line_extension
                
                
                
                if y_n_up is not None and y_n_down is not None:
                    c.p0=[x_n,y_n_up]
                    c.p1=[x_n,y_n_down]

                elif y_n_up is not None:
                    c.p0=[x_n,y_n_up]
                    c.p1=[x_n,y_n]
                else:
                    c.p0=[x_n,y_n]
                    c.p1=[x_n,y_n_down]
                    
                nn=[]
                nn=self.next_grids[x_n]
                if len(nn)<2:
                    if y>y_n and y_n_up!=None :
                        c.p0=[x_n,y_n_up]
                        c.p1=[x_n,y_n]
                    elif y_n_down!=None :
                        c.p0=[x_n,y_n]
                        c.p1=[x_n,y_n_down]
                    
                
                if y_up is not None and y_down is not None:
                    if len(next_grids)==1:
                        c.p3=[x,y_up]
                        c.p2=[x,y_down]
                    else:
                        if y>=y_n: 
                            c.p3=[x,y]
                            c.p2=[x,y_down]
                            
                        else:
                            c.p3=[x,y_up]
                            c.p2=[x,y]    
                
                elif y_up is not None:
                    c.p3=[x,y_up]
                    c.p2=[x,y]   
                elif y_down is not None:
                    c.p3=[x,y]
                    c.p2=[x,y_down]
                    
                    
                if self.points_border_dict[str(c.p0[0])+"-"+str(c.p0[1])] and self.points_border_dict[str(c.p3[0])+"-"+str(c.p3[1])] :
                     c.constrained_adj_up=True
                elif self.points_border_dict[str(c.p0[0])+"-"+str(c.p0[1])] or self.points_border_dict[str(c.p3[0])+"-"+str(c.p3[1])] :
                     c.constrained_adj_half_up=True
                if self.points_border_dict[str(c.p1[0])+"-"+str(c.p1[1])] and self.points_border_dict[str(c.p2[0])+"-"+str(c.p2[1])]:
                     c.constrained_adj_down=True
                elif self.points_border_dict[str(c.p1[0])+"-"+str(c.p1[1])] or self.points_border_dict[str(c.p2[0])+"-"+str(c.p2[1])]:
                     c.constrained_adj_half_down=True
                    
                self.grid.append(c)
            
        ##Add neighbors
        print("Add neighboors")
        for cell in self.grid:
            ii=cell.id[1]
            
            ##Compute center
            cell.center_x=(cell.p0[0]+cell.p1[0]+cell.p2[0]+cell.p3[0])/4.0
            cell.center_y=(cell.p0[1]+cell.p1[1]+cell.p2[1]+cell.p3[1])/4.0
            x=cell.center_x
            y=cell.center_y
            cell.key_index=int((-(x+y)//1+(x-(x//1))*10000000000+(y-(y//1))*10000000000)//1)
            
        for cell in self.grid:
            ii=cell.id[1]
            for cc in self.grid:
                if ii ==cc.id[0] and ii is not None:
                    cell.neighbors.append(cc.key_index)
                    cc.neighbors.append(cell.key_index)
                    
        #Add geometry points of constrained 
        print("Add geometry points")           
        for num,cell in enumerate(self.grid):
            in_cell=False
            if cell.constrained_adj_down:
                p_right_up=cell.p0
                p_right_down=copy.deepcopy(cell.p1)
                p_left_down=copy.deepcopy(cell.p2)
                p_left_up=cell.p3
                
                p_right_down[1]=p_right_down[1]-1
                p_left_down[1]=p_left_down[1]-1
                
                for i in range(len(self.poly_points[len(self.poly_points)-1])-1):
                    p1=self.poly_points[len(self.poly_points)-1][i]
                    p2=self.poly_points[len(self.poly_points)-1][(i+1)%len(self.poly_points[len(self.poly_points)-1])]
                    boolean_right=intersect(p_right_up,p_right_down,p1,p2)
                    boolean_left=intersect(p_left_up,p_left_down,p1,p2)
               
                    if boolean_right and not boolean_left:
                        in_cell=True
                        
                    elif  boolean_left:
                        in_cell=False

                        
                    if in_cell:
                        cell.constrained_geometry.append(p2)
                        
                    if cell.constrained_geometry!=[]:
                        if cell.constrained_geometry[len(cell.constrained_geometry)-1][0]==cell.p2[0] and cell.constrained_geometry[len(cell.constrained_geometry)-1][1]==cell.p2[1]:
                            cell.constrained_geometry.pop()
                        if cell.constrained_geometry[0][0]==cell.p1[0] and cell.constrained_geometry[0][1]==cell.p1[1]:
                            del cell.constrained_geometry[0]
                    
            elif cell.constrained_adj_up:
                p_right_up=copy.deepcopy(cell.p0)
                p_right_down=cell.p1
                p_left_down=cell.p2
                p_left_up=copy.deepcopy(cell.p3)
                
                p_right_up[1]=p_right_up[1]+1
                p_left_up[1]=p_left_up[1]+1
                
                for i in range(len(self.poly_points[len(self.poly_points)-1])-1):
                    p1=self.poly_points[len(self.poly_points)-1][i]
                    p2=self.poly_points[len(self.poly_points)-1][(i+1)%len(self.poly_points[len(self.poly_points)-1])]
                    boolean_right=intersect(p_right_down,p_right_up,p1,p2)
                    boolean_left=intersect(p_left_down,p_left_up,p1,p2)
                    
                    

                    if boolean_right :
                        in_cell=False

                        
                    if (not boolean_right) and  boolean_left:
                        in_cell=True

                        
                    if in_cell:
                        cell.constrained_geometry.append(p2)
                
                if cell.constrained_geometry!=[]:
                    if cell.constrained_geometry[len(cell.constrained_geometry)-1][0]==cell.p0[0] and cell.constrained_geometry[len(cell.constrained_geometry)-1][1]==cell.p0[1]:
                        cell.constrained_geometry.pop()
                    if cell.constrained_geometry[0][0]==cell.p3[0] and cell.constrained_geometry[0][1]==cell.p3[1]:
                        del cell.constrained_geometry[0]
                        
                        
            #Found the most distant point           
            if cell.constrained_adj_down :
                y_lower=min(cell.p1[1], cell.p2[1])
                for p in cell.constrained_geometry:
                    if p[1]<y_lower:
                        y_lower=p[1]
                cell.y_lower=y_lower
                
            elif cell.constrained_adj_up:
                y_max=max(cell.p0[1], cell.p3[1])
                for p in cell.constrained_geometry:
                    if p[1]>y_max:
                        y_max=p[1]
                cell.y_max=y_max