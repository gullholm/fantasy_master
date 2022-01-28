# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 08:21:55 2022

@author: jgull
"""


import getters
import numpy as np
import time
import matplotlib.pyplot as plt
import calculations as calc
import parsers

# Import the data 
n = 20
gk2, def4, mid4, forw2 = calc.createFormation(4, 4, 2, n)

ss =  parsers.calc_p_c_per_part(gk2, def4, mid4, forw2)

points = parsers.parse_formations_points_or_cost(ss[0][0], ss[0][1], ss[0][2], ss[0][3])
costs = parsers.parse_formations_points_or_cost(ss[1][0], ss[1][1], ss[1][2], ss[1][3])

data2 = getters.get_data()
players = getters.get_players_feature(data2)
gk, df, mf, fw = getters.get_diff_pos(players)

defe = np.transpose(calc.nump2(len(df),4))
midf = np.transpose(calc.nump2(len(mf),4))
forw = np.transpose(calc.nump2(len(fw),2))    
glk = np.transpose(calc.nump2(len(gk),1))

seed =123   

forwards = calc.calcindex(forw, fw, 2, n, seed) 
defenders = calc.calcindex(defe, df, 4, n, seed )
midfielders = calc.calcindex(midf, mf, 4, n, seed)
goalkeepers = calc.calcindex(glk, gk, 1, n, seed)

under_cost =  np.argwhere(costs < 700)
#%%
cost_f = np.zeros(under_cost.shape[0])
for i in range(under_cost.shape[0]):
    cost_f[i] = points[under_cost[i][0],under_cost[i][1],under_cost[i][2],under_cost[i][3]]
    
best = np.argmax(cost_f)

sep_ids = [forwards, midfielders, defenders, goalkeepers]

best_team_ids = [x[under_cost[best][i]] for (i,x) in enumerate(sep_ids)]
