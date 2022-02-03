# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 09:08:44 2022

@author: jgull
"""

import getters
import numpy as np
import parsers


data_as = getters.get_data()
#data_es = getters.get_data("eliteserien", "no")
#data_pl = getters.get_data("premierleague", "com")

players = getters.get_players_feature(data_as)


#team = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

#ddd = getters.get_diff_pos(players)

gk, df, mf, fwd = getters.get_diff_pos(players)

dfPoints, dfCost = [], []

dfPoints = [d.get('total_points') for d in df.values()]

#best_team_ids = parsers.get_best_team_from_random(n = 100, cost_limit = 700) 

best_team_ids = [462, 416, 72, 150, 180, 390, 460, 67, 113, 174, 374] # 700

names = getters.get_full_name_team(data_as, best_team_ids)

best_team_ids_values = [players[ids] for ids in best_team_ids]

random_sum_cost = [player['now_cost'] for player in best_team_ids_values] #653
random_sum_points = [player['total_points'] for player in best_team_ids_values] #941


dfPointz = parsers.change_dict_to_2darray(df, "total_points")
mfPointz = parsers.change_dict_to_2darray(mf, "total_points")
fwdPointz = parsers.change_dict_to_2darray(fwd, "total_points")
gkPointz = parsers.change_dict_to_2darray(df, "total_points")
#%%
import pandas as pd
import parsers
import ast
import getters as get
import calculations as calc


all_combs = get.get_cleaned_combs()
under_70 = calc.calc_best_team(all_combs, 700)

data = getters.get_data()
best_team_i = [item for sublist in best_team_ids for item in sublist]
best_team_names = getters.get_full_name_team(data, best_team_i)

best_team_ids_values = [players[ids] for ids in best_team_i]

best_sum_cost = [player['now_cost'] for player in best_team_ids_values] #653
best_sum_points = [player['total_points'] for player in best_team_ids_values] #941





#%%

costs_full = parsers.parse_formations_points_or_cost(gk_costs, df_costs, 
                                                     mf_costs, fw_costs)

under_cost =  np.argwhere(costs_full < 500) 
best = parsers.find_best_team(under_cost, points_full)
sep_ids = [fw_combs['indexes'].values.tolist(),mf_combs['indexes'].values.tolist() 
           , df_combs['indexes'].values.tolist(), gk_combs['indexes'].values.tolist()]
best_team_ids = [x[under_cost[best][i]] for (i,x) in enumerate(sep_ids)]



#%%
data = getters.get_data()
best_team_i = [item for sublist in best_team_ids for item in sublist]
best_team_names = getters.get_full_name_team(data, best_team_i)

best_team_ids_values = [players[ids] for ids in best_team_i]

best_sum_cost = [player['now_cost'] for player in best_team_ids_values] #653
best_sum_points = [player['total_points'] for player in best_team_ids_values] #941

#%%

import pandas as pd
import parsers
import ast
generic = lambda x: ast.literal_eval(x)

conv = {'indexes': generic}

gk_combs = pd.read_csv("data_cleaned/gk.csv", converters = conv)
fw_combs = pd.read_csv("data_cleaned/fw.csv", converters = conv)
df_combs = pd.read_csv("data_cleaned/df.csv", converters = conv)
mf_combs = pd.read_csv("data_cleaned/mf.csv", converters = conv)
gk_combs['indexes'] = gk_combs['indexes'].apply(lambda x: [x])
#%%


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

under_cost =  np.argwhere(costs_full < 700) 

best = parsers.find_best_team(under_cost, points_full)
sep_ids = [fw_combs['indexes'].values.tolist(),mf_combs['indexes'].values.tolist() 
           , df_combs['indexes'].values.tolist(), gk_combs['indexes'].values.tolist()]

best_team_ids = [x[under_cost[best][i]] for (i,x) in enumerate(sep_ids)]


#%%
data = getters.get_data()
best_team_i = [item for sublist in best_team_ids for item in sublist]
best_team_names = getters.get_full_name_team(data, best_team_i)

best_team_ids_values = [players[ids] for ids in best_team_i]

best_sum_cost = [player['now_cost'] for player in best_team_ids_values] #653
best_sum_points = [player['total_points'] for player in best_team_ids_values] #941