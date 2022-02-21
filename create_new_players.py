# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:15:57 2022

@author: jgull
"""

import numpy as np
import getters as get
import matplotlib.pyplot as plt

data_as = get.get_data()
players = get.get_players_feature(data_as)

def parse_players_feature_to_list(players):
    ids, pos, cost, point = [], [], [], []

    for (k,v) in players.items():
#        ids.append(k)
        pos.append(v.get("element_type"))
        cost.append(v.get("now_cost"))
        point.append(v.get("total_points"))

    return(pos,cost,point)

pos, cost, point = parse_players_feature_to_list(players)

rang = np.linspace(min(cost), max(cost), dtype = int,num = 10)

bins = np.digitize(cost, rang)

