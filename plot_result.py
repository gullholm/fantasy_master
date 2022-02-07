# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:53:29 2022

@author: jonat
"""


import pandas as pd
import parsers
import ast
import numpy as np
import getters
generic = lambda x: ast.literal_eval(x)
import matplotlib.pyplot as plt 


conv = {'indexes': generic}



gk_combs = pd.read_csv("data_cleaned/gk.csv", converters = conv)
fw_combs = pd.read_csv("data_cleaned/fw.csv", converters = conv)
df_combs = pd.read_csv("data_cleaned/df.csv", converters = conv)
mf_combs = pd.read_csv("data_cleaned/mf.csv", converters = conv)
gk_combs['indexes'] = gk_combs['indexes'].apply(lambda x: [x])

gk_points = gk_combs['total_points'].values
df_points = df_combs['total_points'].values
mf_points = mf_combs['total_points'].values
fw_points = fw_combs['total_points'].values

all_points=[gk_points, df_points, mf_points, fw_points]
points_full = parsers.parse_formations_points_or_cost(all_points)

gk_costs = gk_combs['now_cost'].values
df_costs = df_combs['now_cost'].values
mf_costs = mf_combs['now_cost'].values
fw_costs = fw_combs['now_cost'].values

all_costs = [gk_costs, df_costs, mf_costs, fw_costs]
costs_full = parsers.parse_formations_points_or_cost(all_costs)

data = getters.get_data()
players = getters.get_players_feature(data)

#%%
def plot_values(max_cost):
    
    under_cost =  np.argwhere(costs_full < max_cost) 

    best = parsers.find_best_team(under_cost, points_full)
    sep_ids = [fw_combs['indexes'].values.tolist(),mf_combs['indexes'].values.tolist() 
               , df_combs['indexes'].values.tolist(), gk_combs['indexes'].values.tolist()]
    
    best_team_ids = [x[under_cost[best][i]] for (i,x) in enumerate(sep_ids)]
    best_team_i = [item for sublist in best_team_ids for item in sublist]    
    best_team_ids_values = [players[ids] for ids in best_team_i]
    #best_team_names = getters.get_full_name_team(data, best_team_i)

    best_sum_cost = [player['now_cost'] for player in best_team_ids_values] 
    best_sum_points = [player['total_points'] for player in best_team_ids_values]
    
    return best_sum_cost, best_sum_points


# In[]
best_costs, best_points = [],[]

for i in range(450, 850, 50):
    costs, points = plot_values(i)
    best_costs.append(costs)
    best_points.append(points)

# In[]    
# Plots
plt.plot(sorted(best_costs[7]),'o')