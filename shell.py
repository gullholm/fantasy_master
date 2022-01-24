# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 09:08:44 2022

@author: jgull
"""

import getters

data = getters.get_data()

players = getters.get_players_feature(data)


team = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

#ddd = getters.get_diff_pos(players)

gk, d, mf, fwd = getters.get_diff_pos(players)