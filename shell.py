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

#best_team_under_75 = parsers.get_best_team_from_random(n = 100) 

best_team_under_75 = [300, 8, 76, 174, 407, 72, 150, 180, 390, 462, 416]

names = getters.get_full_name_team(data_as, team)

random_team_values = [players[ids] for ids in team]

random_sum_cost = [player['now_cost'] for player in random_team_values] #653
random_sum_points = [player['total_points'] for player in random_team_values] #941


dfPointz = parsers.change_dict_to_2darray(df, "total_points")
mfPointz = parsers.change_dict_to_2darray(mf, "total_points")
fwdPointz = parsers.change_dict_to_2darray(fwd, "total_points")
gkPointz = parsers.change_dict_to_2darray(df, "total_points")

