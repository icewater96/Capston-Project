# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import geopy
from geopy.distance import vincenty
from geopy.distance import great_circle
from random import *
import math
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation

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

d = pd.read_table( "..\\Capstone Project\\data\\_dcebfb2135a2bf5a6392493bd61aba22_detroit-demolition-permits.tsv", low_memory = False )

d_len = len(d)

address = []
lon = []
lat = []
combined = []

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
    combined.append( (temp_lon_lat[1], temp_lon_lat[0]) )
    
d['address'] = address
d['LON'] = lon
d['LAT'] = lat
d['LAT, LON'] = combined

d['Lat (radian)'] = map( math.radians, d['LAT'])
d['Lon (radian)'] = map( math.radians, d['LON'])

if False:
    d.to_csv( 'temp Demolition.csv' )


#%% Read Blight file
b = pd.read_csv( "..\\Capstone Project\\data\\_97bd1c1e5df9537bb13398c9898deed7_detroit-blight-violations.csv", low_memory = False )

b_len = len(b)

address = []
lon = []
lat = []
combined = []

for i in range(b_len):
    if i % 100 == 0:
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
    combined.append( (temp_lon_lat[1], temp_lon_lat[0]) )
    
b['address'] = address
b['LON'] = lon
b['LAT'] = lat
b['LAT, LON'] = combined

b['Lat (radian)'] = map( math.radians, b['LAT'])
b['Lon (radian)'] = map( math.radians, b['LON'])

b['ViolationCategory'] = b['ViolationCategory'].astype('string')

b['Fee'] = [''] * len(b)

JudgmentAmt = b['JudgmentAmt'].copy()
JudgmentAmt = JudgmentAmt.str.replace( '$', '' )
JudgmentAmt = pd.to_numeric( JudgmentAmt )
b.ix[ JudgmentAmt < 200, 'Fee' ] = '< $200'
b.ix[ (JudgmentAmt >= 200) & (JudgmentAmt < 500), 'Fee' ] = '< $500'
b.ix[ JudgmentAmt >= 500, 'Fee' ] = '> $500'

if False:
    b.to_csv( 'temp Blight.csv' )

#%% 311 
c = pd.read_csv( "..\\Capstone Project\\data\\_dcebfb2135a2bf5a6392493bd61aba22_detroit-311.csv", low_memory = False )
c['Lat (radian)'] = map( math.radians, c['lat'])
c['Lon (radian)'] = map( math.radians, c['lng'])


crime = pd.read_csv( "..\\Capstone Project\\data\\_dcebfb2135a2bf5a6392493bd61aba22_detroit-crime.csv", low_memory = False )
crime['Lat (radian)'] = map( math.radians, crime['LAT'])
crime['Lon (radian)'] = map( math.radians, crime['LON'])

#%% Select good positive from demolition csv
# This is positive address

if False:
    selected_lat = []
    selected_lon = []
    selected_min_distance = []
    
    good_filter_1 = abs( d['LAT'] - 42.331681138 ) > 1e-8
    good_filter_2 = abs( d['LAT'] - 42.395048155 ) > 1e-8
    good_filter_3 = abs( d['LAT'] - 42.331683222 ) > 1e-8
    good_filter_4 = ( d['LAT'] > 0 ) & d['LAT'] < 45
    
    good_filter_5 = abs( d['LON'] + 83.047996037 ) > 1e-8
    good_filter_6 = abs( d['LON'] + 83.047999033 ) > 1e-8
    good_filter_7 = abs( d['LON'] + 83.225058015 ) > 1e-8
    good_filter_8 = d['LON'] < 0
    
    good_filter = good_filter_1 & good_filter_2 & good_filter_3 & good_filter_4 & good_filter_5 & good_filter_6 & good_filter_7 & good_filter_8
    selected_d = d.ix[ good_filter, : ]
    selected_d.index = ['']* len( selected_d )
    
    temp_len = len( selected_d )
    for i in range(temp_len):
        print i 
        temp_min_distance = 1e8
        for j in range(i+1, temp_len):
            temp_distance = great_circle( selected_d['LAT, LON'][i], selected_d['LAT, LON'][j]).meters
            if temp_distance < temp_min_distance:
                temp_min_distance = temp_distance
                
        if temp_min_distance > 2.0:   # 2 meters
            selected_lat.append( selected_d['LAT'][i])
            selected_lon.append( selected_d['LON'][i])
            selected_min_distance.append( temp_min_distance )
    
    selected_df = pd.DataFrame( { 'LAT': selected_lat, 'LON': selected_lon, 'Min Distance': selected_min_distance})	

    if False:
        selected_df.to_csv( 'Positive Addresses from Demolition.csv' )
else:
    selected_df = pd.read_csv( 'Positive Addresses from Demolition.csv' )       
    
positive_address = selected_df    
    
#%% Search in blight data for negative addresses
# Negative sample is defined as min_distance to all positive addresses < 2 meters

if False:
    good_filter_1 = abs( b['LAT'] - 42.331681138 ) > 1e-8
    good_filter_2 = abs( b['LAT'] - 42.395048155 ) > 1e-8
    good_filter_3 = abs( b['LAT'] - 42.331683222 ) > 1e-8
    good_filter_4 = ( b['LAT'] > 0 ) & b['LAT'] < 45
    
    good_filter_5 = abs( b['LON'] + 83.047996037 ) > 1e-8
    good_filter_6 = abs( b['LON'] + 83.047999033 ) > 1e-8
    good_filter_7 = abs( b['LON'] + 83.225058015 ) > 1e-8
    good_filter_8 = b['LON'] < 0
    
    good_filter = good_filter_1 & good_filter_2 & good_filter_3 & good_filter_4 & good_filter_5 & good_filter_6 & good_filter_7 & good_filter_8
    
    b_subset = b.ix[good_filter, :]
    
    negative_lat = []
    negative_lon = []
    negative_min_distance = []
    
    shuffled_index = list( b_subset.index )
    shuffle( shuffled_index )      # In-place shuffling
    
    counter = 0
    for i in range( len(shuffled_index)):
        print i
        
        b_index = shuffled_index[i]
        print ' b index = ', b_index
        
        temp_min_distance = 1e8
        
        # go through each positive address
        for positive_index in range( len( positive_address ) ):
            temp_distance = great_circle( b_subset['LAT, LON'][b_index], ( positive_address['LAT'][positive_index], 
                                                                    positive_address['LON'][positive_index])).meters
            if temp_distance < temp_min_distance:
                temp_min_distance = temp_distance
        
        if temp_min_distance > 2.0:  # 2 meters
            counter += 1
            negative_lat.append( b_subset['LAT'][b_index])                                                       
            negative_lon.append( b_subset['LON'][b_index])
            negative_min_distance.append( temp_min_distance )
            print '   counter = ', counter
            print '   min distance = ', temp_min_distance
            
        if counter > 7000:
            break
            
    # Need to remove duplicates       
    
    neg_lat = []
    neg_lon = []
    neg_min_distance = []
    
    counter = 0
    for i in range( len( negative_lat )):
        print i
        temp_min_distance = 1e8
        for j in range( i+1, len(negative_lat) ):
            temp_distance = great_circle( (negative_lat[i], negative_lon[i]), 
                                          (negative_lat[j], negative_lon[j])  ).meters
            if temp_distance < temp_min_distance:
                temp_min_distance = temp_distance
    
        if temp_min_distance > 2.0:
            counter += 1
            neg_lat.append( negative_lat[i] )
            neg_lon.append( negative_lon[i])
            neg_min_distance.append( temp_min_distance )
    
            print '   counter = ', counter
            print '   min distance = ', temp_min_distance
            
            
    neg_df = pd.DataFrame( { 'LAT': neg_lat, 'LON': neg_lon, 'Min Distance': neg_min_distance})	    
    
    if False:
        neg_df.to_csv( 'Negative Address.csv' )

else:
    neg_df = pd.read_csv( 'Negative Address.csv' )
        
        
        
#%% Extract features
positive_address['Blight Label'] = [1] * len( positive_address)
neg_df['Blight Label'] = [0] * len( neg_df )

feature = pd.concat([positive_address, neg_df], ignore_index = True)
del feature['Unnamed: 0']
feature.index = range(len(feature))

def one_row_feature( lat, lon, column_tuple, feature_names ):
    # Return a 1-dimensional DataFrame or Series to be merged with other rows. 
    temp_cos = map( math.cos, column_tuple['Lat (radian)'] )
    x = 6373.0 * 1000 * abs( lon * 3.1415926 / 180 - column_tuple['Lon (radian)']) * temp_cos  # In meters
    y = 6373.0 * 1000 * abs( lat * 3.1415926 / 180 - column_tuple['Lat (radian)'])
    
    temp = pd.DataFrame( {'x': x, 'y': y} )
    distance = temp.max( axis = 1 )
    
    temp_list_pool = []
    for feature_name in feature_names:
        # 20 meters
        #if sum( distance < 20.0 ) > 1:

        temp_data_1 = pd.crosstab( '', column_tuple.ix[ distance < 20.0, feature_name ] )
        try: 
            temp_column_1 = list( temp_data_1.columns )
            temp_column_1 = [ '20m ' + x for x in temp_column_1 ]
            temp_data_1.columns = temp_column_1
            temp_list_pool.append( temp_data_1 )
        except:
            pass
            
        
        # 200 Meteirs    
        #if sum( distance < 200.0 ) > 1:
    
        temp_data_2 = pd.crosstab( '', column_tuple.ix[ distance < 200.0, feature_name ] )
        try:
            temp_column_2 = list( temp_data_2.columns )
            temp_column_2 = [ '200m ' + x for x in temp_column_2 ]
            temp_data_2.columns = temp_column_2
            temp_list_pool.append( temp_data_2 ) 
        except:
            pass            
 
        # 2000 Meteirs    
        #if sum( distance < 2000.0 ) > 1:
          
        temp_data_3 = pd.crosstab( '', column_tuple.ix[ distance < 2000.0, feature_name ] )
        try:
            temp_column_3 = list( temp_data_3.columns )
            temp_column_3 = [ '2000m ' + x for x in temp_column_3 ]
            temp_data_3.columns = temp_column_3
            temp_list_pool.append( temp_data_3 )
        except:
            pass
    
    temp_data = pd.concat( temp_list_pool, axis = 1 )   
        
    return temp_data
    
   

temp_feature = []
for i in range(len(feature)):
#for i in range(200):
    if i%10 == 0:
        print i
    temp_1 = one_row_feature( feature['LAT'][i], feature['LON'][i], b, ['ViolationCode', 'PaymentStatus', 'ViolationCategory', 'Fee'] )
    
    temp_2 = one_row_feature( feature['LAT'][i], feature['LON'][i], c, ['issue_type'] )
    
    temp_3 = one_row_feature( feature['LAT'][i], feature['LON'][i], crime, ['CATEGORY'] )    
    
    temp_all =  pd.concat( [temp_1, temp_2, temp_3], axis = 1 )
    temp_all['i'] = i
    temp_feature.append( temp_all )
    
    
feature_set_1 = pd.concat(temp_feature)    
feature_set_1.fillna( 0, inplace = True)

truth_label = feature['Blight Label'].copy()  # Truth table

if False:
    feature_set_1.to_csv( 'Features from Blight 311 Crime.csv' )


#%% Cross validation random forest
all_features = feature_set_1.copy()
del all_features['i']

cv_set = StratifiedKFold( truth_label, 5, shuffle = True, random_state = 0)


if True:
    n_estimators = 2000
    max_depth = 20
    min_sample_split = 20
    min_sample_leaf = 5
    n_jobs = -1
    max_features = 'auto'

    clf = RandomForestClassifier(n_estimators = n_estimators, 
                                 max_depth = max_depth,
                                 min_samples_split = min_sample_split, 
                                 min_samples_leaf = min_sample_leaf,
                                 max_features = max_features,
                                 n_jobs=n_jobs, random_state=0, verbose=1 )
else:
    clf = ExtraTreesClassifier(n_estimators = n_estimators, 
                                 max_depth = max_depth,
                                 min_samples_split = min_sample_split, 
                                 min_samples_leaf = min_sample_leaf,
                                 n_jobs=n_jobs, random_state=0, verbose=1 )
    
#clf.fit( all_features.as_matrix(), truth_label.as_matrix())                             

X = all_features.as_matrix()
y = truth_label.as_matrix()
cv_score = cross_validation.cross_val_score( clf, X, y, cv = 5)
    
print cv_score.mean()


# For single feature
clf.fit( X, y )
f = pd.DataFrame( {'Importance': clf.feature_importances_, 'Feature': all_features.columns })
f.sort_values( ['Importance'], inplace = True )

clf_single = RandomForestClassifier(n_estimators = n_estimators, 
                                 max_depth = max_depth,
                                 min_samples_split = min_sample_split, 
                                 min_samples_leaf = min_sample_leaf,
                                 max_features = max_features,
                                 n_jobs=n_jobs, random_state=0, verbose=1 )


f.index = [''] * len( f )

n_top_feature = 1
feature_list = f['Feature'][-n_top_feature :]
#clf_single.fit( all_features[feature_list].as_matrix(), truth_label.as_matrix() )
#print clf_single.score( all_features[feature_list].as_matrix(), truth_label.as_matrix() )
cv_single_score = cross_validation.cross_val_score( clf_single, all_features[feature_list].as_matrix(), truth_label.as_matrix(), cv = 5)
print cv_single_score.mean()

clf_all = RandomForestClassifier(n_estimators = n_estimators, 
                                 max_depth = max_depth,
                                 min_samples_split = min_sample_split, 
                                 min_samples_leaf = min_sample_leaf,
                                 max_features = max_features,
                                 n_jobs=n_jobs, random_state=0, verbose=1 )
clf_all.fit( X, y )
print clf_all.score( X, y )



#%% Calculate address-to-address distance
raise Exception( 'JL stop')

d_distance= []
point_1_lat = []
point_1_lon = []
point_2_lat = []
point_2_lon = []

d_len = len( d )
for i in range( 1, d_len, 2 ):
    print i
    for j in range( i+1, d_len, 2 ):
        temp = great_circle( d['LAT, LON'][i], d['LAT, LON'][j]).meters
        d_distance.append( temp )        
        point_1_lat.append( d['LAT'][i] ) 
        point_1_lon.append( d['LON'][i] ) 
        point_2_lat.append( d['LAT'][j] ) 
        point_2_lon.append( d['LON'][j] ) 

d_distance_df = pd.DataFrame( { 'Distance': d_distance, 'LAT 1': point_1_lat, 'LON 1': point_1_lon, 'LAT 2': point_2_lat, 'LON 2': point_2_lon })

if False:
    d_distance_df.to_csv( 'temp Distance betweeen Demolition.csv' )

