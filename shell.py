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

#random_team = parsers.get_best_team_from_random(n = 100, cost_limit = 700) 

random_team = [462, 416, 72, 150, 180, 390, 460, 67, 113, 174, 374] # 700

names = getters.get_full_name_team(data_as, random_team)

random_team_values = [players[ids] for ids in random_team]

random_sum_cost = [player['now_cost'] for player in random_team_values] #653
random_sum_points = [player['total_points'] for player in random_team_values] #941


dfPointz = parsers.change_dict_to_2darray(df, "total_points")
mfPointz = parsers.change_dict_to_2darray(mf, "total_points")
fwdPointz = parsers.change_dict_to_2darray(fwd, "total_points")
gkPointz = parsers.change_dict_to_2darray(df, "total_points")
#%%
import pandas as pd
import parsers
gk_combs = pd.read_csv("data_cleaned/1_goalkeeper.csv")
fw_combs = pd.read_csv("data_cleaned/2_forwards.csv")
df_combs = pd.read_csv("data_cleaned/4_defenders.csv")
mf_combs = pd.read_csv("data_cleaned/4_midfielders.csv")

gk_points = gk_combs['total_points'].values
df_points = df_combs['total_points'].values
mf_points = mf_combs['total_points'].values
fw_points = fw_combs['total_points'].values

points_full = parsers.parse_formations_points_or_cost(gk_points, 
                                                      df_points, mf_points, fw_points)

gk_costs = gk_combs['now_cost'].values
df_costs = df_combs['now_cost'].values
mf_costs = mf_combs['now_cost'].values
fw_costs = fw_combs['now_cost'].values


costs_full = parsers.parse_formations_points_or_cost(costs_comb[0],
                                                     costs_comb[1], 
                                                     costs_comb[2], costs_comb[3])
