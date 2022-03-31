# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:53:10 2022

@author: jgull
"""

import pandas as pd
import sklearn.linear_model as sklin
import matplotlib.pyplot as plt
from collections import Counter
import random
import getters
import numpy as np

season = 1819
loc = "data/pl_csv/players_raw_"
def create_new_players(loc, season):
    playerspldata = getters.get_players_feature_pl(loc,
                                               season, ['element_type', 'now_cost', 'total_points', 'id'])

    costs = [v.get("now_cost") for (k,v) in playerspldata.items()]
    points = [v.get("total_points") for (k,v) in playerspldata.items()]
    pos = [v.get("element_type") for (k,v) in playerspldata.items()]
    
    counts = Counter()
    counts.update({x:0 for x in range(min(costs),max(costs)+1)})
    counts.update(costs)
    max_counts = counts.most_common(1)[0][1]
    new_players = playerspldata.copy() 
    count_list = list(counts.values()) 
    costs_np = np.array(costs).reshape(-1,1)
    points_np = np.array(points).reshape(-1,1)
    linreg = sklin.LinearRegression()
    linreg.fit(costs_np, points_np)
    
    for i, (k,v) in enumerate(counts.items()):
        
        if (v > 0):
            count_before = 0
            inds = [i for (i,x) in enumerate(costs) if x == k]
        else:
            count_after = [j for (j,x) in enumerate(count_list[i:]) if x > 0 ][0]
            count_before += 1
            inds = [j for (j,x) in enumerate(costs) if x == k-count_before] # Find the players at closest distance wrt cost
            inds.extend([j for (j,x) in enumerate(costs) if x == k+count_after])
            

        for i in range(max_counts-v):
            copy_ind = random.sample(inds, 1)[0]
            perc = 0.2
            new_points = np.round(linreg.predict(np.array(k).reshape(-1,1))) + + random.randint(-round(perc*points[copy_ind]), round(perc*points[copy_ind]))   
       
        #print(points[copy_ind])
     #      print(new_points)
            temp = {len(new_players)+1:{"element_type" : pos[copy_ind], "now_cost" : k, "total_points": int(new_points), "id" : len(new_players)+1}}
            new_players.update(temp)
    return(new_players)


news = create_new_players(loc,season)

pl = pd.DataFrame.from_dict(news, orient = "index")

pl.to_csv("data/pl_csv/players_incnew_lin_" + str(season) + ".csv", index = False)
