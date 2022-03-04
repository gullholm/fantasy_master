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

one = pd.read_csv("data_cleaned/pl/incnew/1819/[4, 4, 2].csv", converters = conv)
#%%
def filter_df(df, lwr, upper):
    df = df[df['cost'] <= upper]
    df_new = df[df['cost'] >= lwr]
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

def is_diverse_ed2(team_id, cost_list):
    team_cost = get.get_cost_team(cost_list, team_id)
#    print(team_id)
    team_cost.sort()
#    print(team_cost)
#    a = det_a(full_data, team_id

    a = min(team_cost)
    b= max(team_cost)
#    print(a)
    theory_int = np.linspace(a, b, 11, dtype=int).tolist()
#    print(theory_int)
    c = int(theory_int[1]-theory_int[0])
    theory_int_l = [x - math.ceil(c/2) for x in theory_int]
    theory_int_h = [x + math.ceil(c/2) for x in theory_int]
    counts = [0]*11
#    print(c)
#    print(theory_int_l)
#    print(theory_int_h)
#    save_empty_indices = []
#    print(theory_int_l)
#    print(team_cost)
#    print(theory_int_l)
#    print(theory_int_h)    
    
    for re in team_cost:
        
        for i,(low,up) in enumerate(zip(theory_int_l, theory_int_h)):
            if(re >= low and re <= up):
                counts[i] += 1
#    print(counts)

    
    if(counts.count(0) < 3): # zeros implicates amount that interval doesn't cover
        return(1)
    else:
        return(0)

def flatten(l):
    flattened = []
    for sublist in l:
        flattened.extend(sublist)
    return flattened
#%%
#all_ids = list(flatten_all_ids(one["indexes"].to_list()))
#a = det_a(data, all_ids)

import random
import calculations as calc
import getters

#inter = thry_interval(a, 700)
#players = pd.read_csv("data/pl_csv/players_incnew_1819.csv")
#playerspl = players.to_dict('index')
playerspldata = getters.get_players_feature_pl("data/pl_csv/players_incnew_", 1819)
cost_list = calc.createCostList(playerspldata, False)
budget = 700
ones = filter_df(one, budget-50, budget)
all_teams = ones["indexes"].to_list()
#all_teams = random.sample(all_teams, 50)
#%%
is_dev_or_not = [is_diverse_ed2(team_id, cost_list) for team_id in all_teams]
print(sum(is_dev_or_not))

#%%
print(sum(is_dev_or_not))
indexes_div = [i for (i,x) in enumerate(is_dev_or_not) if x==1]
tot_points = []
tot_cost = []
for ind in indexes_div:
    tot_points.append(ones.iloc[ind]['points_total'])
    tot_cost.append(ones.iloc[ind]['cost'])
    
print((sum(tot_cost)/len(tot_cost)/ ones['cost'].mean()))
print((sum(tot_points)/len(tot_points))/ ones['points_total'].mean())


#%%