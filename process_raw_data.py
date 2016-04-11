# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np


def get_lon_lat( input_string ):
    temp = input_string.replace( "(", "")
    temp = temp.replace( ")", "" )
    
    temp = temp.split( ',' )
    
    if len(temp)> 1:
        lon = float( temp[1] )
        lat = float( temp[0] )
    else:
        lon = 0
        lat = 0
    
    return (lon, lat)

#%% clean demolition permit file

d = pd.read_table( "..\\Capstone Project\\data\\_dcebfb2135a2bf5a6392493bd61aba22_detroit-demolition-permits.tsv" )

d_len = len(d)

address = []
lon = []
lat = []

for i in range(d_len):
    print i
    if type( d['site_location'][i] ) == float:
        temp_lon_lat = (0, 0)
        temp_address = ''
    else:
        temp = d['site_location'][i].split( '\n' )
        
        if len(temp) == 1:
            temp_lon_lat = get_lon_lat( temp[0] )
            temp_address = ''
        else:
            temp_lon_lat = get_lon_lat( temp[2])
            temp_address = temp[0]
            
    address.append( temp_address )
    lon.append( temp_lon_lat[0])
    lat.append( temp_lon_lat[1])
    
d['address'] = address
d['LON'] = lon
d['LAT'] = lat

if False:
    d.to_csv( 'temp Demolition.csv' )


#%% Blight file


b = pd.read_csv( "..\\Capstone Project\\data\\_97bd1c1e5df9537bb13398c9898deed7_detroit-blight-violations.csv", low_memory = False )

b_len = len(b)

address = []
lon = []
lat = []

for i in range(b_len):
    print i
    if type( b['ViolationAddress'][i] ) == float:
        temp_lon_lat = (0, 0)
        temp_address = ''
    else:
        temp = b['ViolationAddress'][i].split( '\n' )
        
        if len(temp) == 1:
            temp_lon_lat = get_lon_lat( temp[0] )
            temp_address = ''
        else:
            temp_lon_lat = get_lon_lat( temp[2])
            temp_address = temp[0]
            
    address.append( temp_address )
    lon.append( temp_lon_lat[0])
    lat.append( temp_lon_lat[1])

    
b['address'] = address
b['LON'] = lon
b['LAT'] = lat

if False:
    b.to_csv( 'temp Blight.csv' )



