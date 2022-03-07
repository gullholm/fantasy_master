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

one = pd.read_csv("data_cleaned/pl/noexp/1617/[4, 5, 1].csv", converters = conv)
#%%%
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

def is_diverse_ed2(playersdata, team_ids):
    team_cost = get.get_cost_team(playersdata, team_ids)
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
    theory_int_l = [x - math.ceil(c) for x in theory_int]
    theory_int_h = [x + math.ceil(c) for x in theory_int]
    counts = [0]*11
#    print(c)
#    print(theory_int_l)
#    print(theory_int_h)
#    save_empty_indices = []
#    print(theory_int_l) 
#    print(team_cost)
#    print(theory_int)
#    print(theory_int_h)    
    
    for re in team_cost:
        
        for i,(low,up) in enumerate(zip(theory_int_l, theory_int_h)):
            if(re >= low and re <= up):
                counts[i] += 1
#    print(counts)

    
    if(counts.count(0) < 3): # zeros implicates amount that interval doesn't cover
        #return(1)
        return(0)
    else:
        #return(0)
        return(1)

def flatten(l):
    flattened = []
    for sublist in l:
        flattened.extend(sublist)
    return flattened
import matplotlib.pyplot as plt
def rmse(actual, predicted):
    return np.sqrt(np.square(np.subtract(np.array(actual), np.array(predicted))).mean())

def testIfLinear(data, budget):
    x=range(1,12)
    print(budget)
    low = data[0]
    high = data[len(data)-1]
    for degree in range(1,4):
        poly= np.polyfit(x,data,degree)
        ypred = np.polyval(poly,x)
        plt.plot(x, ypred)
        print('RMSE deg ' + str(degree) +  ': ' + str(rmse(data,ypred)))
        #print('RMSPE deg ' + str(degree) +  ': ' + str(rmspe(data,ypred)))

    plt.title("mean for: " + str(budget))
    plt.xlabel("Player")
    plt.ylabel("Normalized cost")
    plt.plot(x, data, 'o')
    plt.legend(["Linear", "Quadtratic", "Third degree polynomial", "Data"])
    plt.show()
     

#%%%
#all_ids = list(flatten_all_ids(one["indexes"].to_list()))
#a = det_a(data, all_ids)

import random
import calculations as calc
import getters

#inter = thry_interval(a, 700)
#players = pd.read_csv("data/pl_csv/players_incnew_1819.csv")
#playerspl = players.to_dict('index')
playerspldata = getters.get_players_feature_pl("data/pl_csv/players_noexp_0.1_", 1617)
cost_list = calc.createCostList(playerspldata, False)
#%%

budget =500
ones = filter_df(one, 0, budget)
ones.sort_values(by ="points_total", inplace = True, ascending = False)

#ones = ones.sample(n = 50)
all_teams = ones["indexes"].to_list()
#all_teams_cost_list = [get.get_cost_team(cost_list, team_id) for team_id in all_teams]
#testIfLinear(all_teams_cost_list[0], budget)


#all_teams = random.sample(all_teams, 50)

#%%
from collections import Counter
is_dev_or_not = [is_diverse_ed2(playerspldata, team_id) for team_id in all_teams]

print(sum([is_dev_or_not[x][0] for x in range(len(is_dev_or_not))]))
print(sum(is_dev_or_not))
enum = [[i for (i,x) in enumerate(is_dev_or_not[y][1]) if x == 0]  for y in range(len(is_dev_or_not))]
cunt = Counter(flatten(enum)).most_common()
#%%
#print(sum(is_dev_or_not))
indexes_div = [i for (i,x) in enumerate(is_dev_or_not) if x==1]
tot_points = []
tot_cost = []
for ind in indexes_div:
    tot_points.append(ones.iloc[ind]['points_total'])
    tot_cost.append(ones.iloc[ind]['cost'])
    
print((sum(tot_cost)/len(tot_cost)/ ones['cost'].mean()))
print((sum(tot_points)/len(tot_points))/ ones['points_total'].mean())


#%%