# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 18:00:19 2022

@author: jgull
"""

import getters
from collections import Counter
import random
import pandas as pd

season = 1617
playerspldata = getters.get_players_feature_pl("data/pl_csv/players_raw_",
                                               season, ['element_type', 'now_cost', 'total_points', 'id'])

costs = [v.get("now_cost") for (k,v) in playerspldata.items()]
points = [v.get("total_points") for (k,v) in playerspldata.items()]
pos = [v.get("element_type") for (k,v) in playerspldata.items()]

counts = Counter()
counts.update({x:0 for x in range(min(costs),max(costs)+1)})
counts.update(costs)
keys = list(counts.keys())
max_counts = counts.most_common(1)[0][1]
perc = 0.1
new_players = playerspldata.copy() 
count_list = list(counts.values()) 
for i, (k,v) in enumerate(counts.items()):
    if( v > 0 ): # If we have players with that cost
        count_before = 0
        inds = [i for (i,x) in enumerate(costs) if x == k]
        #print(max_counts-v)
        for i in range(max_counts-v):
            copy_ind = random.sample(inds, 1)[0]

            new_points = points[copy_ind] + random.randint(-round(perc*points[copy_ind]), round(perc*points[copy_ind]))   
#            print(points[copy_ind])
 #           print(new_points)
            temp = {len(new_players)+1:{"element_type" : pos[copy_ind], "now_cost" : k, "total_points": new_points, "id" : len(new_players)+1}}
            new_players.update(temp)
    else: # If we have no players at that cost
        count_before += 1 # Keep order of distance (in cost) to where there are players
        count_after = [j for (j,x) in enumerate(count_list[i:]) if x > 0 ][0]
        inds_before = [j for (j,x) in enumerate(costs) if x == k-count_before] # Find the players at closest distance wrt cost
        inds_after = [j for (j,x) in enumerate(costs) if x == k+count_after]
        print(k-count_before)
        print(inds_before)
        w_before = count_after / (count_before+count_after)
        w_after = count_before / (count_before+count_after)
        
        for i in range(max_counts):
            copy_ind_before = random.sample(inds_before, 1)[0]
            copy_ind_after = random.sample(inds_after, 1)[0]
            
            new_points_before = points[copy_ind_before] + random.randint(-round(perc*points[copy_ind_before]), round(perc*points[copy_ind_before]))
            new_points_after = points[copy_ind_after] + random.randint(-round(perc*points[copy_ind_after]), round(perc*points[copy_ind_after]))
            
            weighted_points = round(w_before * new_points_before + w_after * new_points_after)
            temp = {len(new_players)+1:{"element_type" : pos[random.sample([copy_ind_before,copy_ind_after],1)[0]],
                                                                           "now_cost" : k, "total_points": weighted_points,
                                                                           "id" : len(new_players)+1}}
            new_players.update(temp)

            

costs2 = [v.get("now_cost") for (k,v) in new_players.items()]
points2 = [v.get("total_points") for (k,v) in new_players.items()]

pl = pd.DataFrame.from_dict(new_players, orient = "index")

pl.to_csv("data/pl_csv/players_incnew_" + str(season) + ".csv")

counts2 = Counter(costs2)
