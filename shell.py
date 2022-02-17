# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 09:08:44 2022

@author: jgull
"""
import numpy as np
import getters

data = getters.get_data()

players = getters.get_players_feature(data)


team = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

#ddd = getters.get_diff_pos(players)

gk, d, mf, fwd = getters.get_diff_pos(players)

team_names = getters.get_cost_team(data, team)


ll= [1,2,3]

ee = np.array(ll)

kk = [10,15,20,35]
ddd = np.array(kk)

print(np.sum(ddd[ll]))