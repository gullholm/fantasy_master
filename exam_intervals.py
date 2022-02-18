#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:38:11 2022

@author: josef
"""

import pandas as pd
import numpy as np
import getters as get
import ast
generic = lambda x: ast.literal_eval(x)
conv = {'indexes': generic}

data = get.get_data()

one = pd.read_csv("data_cleaned/as/[3, 4, 3].csv", converters = conv)
#%%
def filter_df(df, lwr, upper):
    df = df[df['cost'] < upper]
    df_new = df[df['cost'] > lwr]
    return(df_new)

def lower_bound(df_list):
    return 0

def flatten_all_ids(all_teams_id):
    return(set([item for sublist in all_teams_id for item in sublist]))

def det_a(data, all_ids):
    return(min(get.get_cost_team(data,all_ids)))

def thry_interval(a,upper, c = 2):
    inter = np.linspace(a+c, (2*upper/11)-a, 11, dtype = int)
    return(inter.tolist())

def is_diverse(team_id, full_data, budget, c = 5):
    team_cost = get.get_cost_team(full_data, team_id)
    team_cost.sort()
    a = det_a(full_data, team_id)
    theory_int = thry_interval(a, budget)
    oks = 0
    for (th, re) in zip(theory_int, team_cost):
        if (re < th + c) and (re > th - c):
            oks += 1
    if(oks<9):
        return(0)
    else:
        return(1)
import math

def is_diverse_ed2(team_id, full_data, budget, c = 1):
    team_cost = get.get_cost_team(full_data, team_id)
    team_cost.sort()
    a = det_a(full_data, team_id)
#    print(a)
    theory_int = thry_interval(a, budget)
    c = int(math.ceil((theory_int[1]-theory_int[0])/2))
#    print(c)
    theory_int_l = [x - c for x in theory_int]
    theory_int_h = [x + c for x in theory_int]
#    print(theory_int)
#    print(team_cost)
    counts = [0]*11
    for re in team_cost:
        
        for i,(low,up) in enumerate(zip(theory_int_l, theory_int_h)):
            if(re >= low and re <= up):
                counts[i] += 1
#    print(counts)
    if(counts.count(0) < 3): # zeros implicates amount that interval doesn't cover
        return(1)
    else:
        return(0)


#%%
#all_ids = list(flatten_all_ids(one["indexes"].to_list()))

#a = det_a(data, all_ids)
import random
#inter = thry_interval(a, 700)
ones = filter_df(one, 650, 700)
all_teams = ones["indexes"].to_list()
#all_teams = random.sample(all_teams, 1000)
is_dev_or_not = [is_diverse_ed2(team_id, data, 700) for team_id in all_teams]
print(sum(is_dev_or_not))
indexes_div = [i for (i,x) in enumerate(is_dev_or_not) if x==1]
tot_points = []
tot_cost = []
for ind in indexes_div:
    tot_points.append(ones.iloc[ind]['points_total'])
    tot_cost.append(ones.iloc[ind]['cost'])
    
print(sum(tot_cost)/len(tot_cost))
print(sum(tot_points)/len(tot_cost))
print(ones['points_total'].mean())
print(ones['cost'].mean())
