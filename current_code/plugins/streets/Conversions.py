# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 16:08:32 2021

@author: nipat
"""
import math

class Point:
    def __init__(self):
        self.x=None
        self.y=None
        self.lat=None
        self.lon=None
        



######################
##Transformations Mercator projection
##Equations from OSM2World github repo 
## https://github.com/tordanik/OSM2World
######################    
class Conversions:
    earth_circumference= 40075016.686
    
    def __init__(self,lat,lon):

        self.origin_lat=lat
        self.origin_lon=lon
        self.origin_x=self.lon2x(self.origin_lon)
        self.origin_y=self.lat2y(self.origin_lat)
        
        self.scale=math.cos(math.radians(lat))*40075016.686
    
    def lon2x(self,lon):
        return (lon+180.0)/360.0
    
    def lat2y(self,lat):
        return math.log((1+math.sin(math.radians(lat)))/(1-math.sin(math.radians(lat))))/(4*math.pi) + 0.5
    
    def geodetic2cartesian(self,lat,lon):
        x=self.scale*(self.lon2x(lon)-self.origin_x)
        y=self.scale*(self.lat2y(lat)-self.origin_y)
        return x,y
    
    def x2lon(self,x):
        return 360.0 * (x - 0.5)
    
    def y2lat(self,y):
        return 360.0 * math.atan(math.exp((y - 0.5) * (2.0 * math.pi))) / math.pi - 90.0
    
    def cartesian2geodetic(self,x,y):
        lon=self.x2lon(x  / self.scale+ self.origin_x)
        lat=self.y2lat(y/ self.scale + self.origin_y )
        return lat,lon

# =============================================================================
# con=Conversions(38.24455,21.736)
# x,y=con.geodetic2cartesian(38.2462420,21.7350847)
# print(x)
# print(y)
# lat,lon=con.cartesian2geodetic(x,y)
# print(lat)
# print(lon)
# =============================================================================
