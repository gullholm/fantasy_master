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
import random
import calculations as calc
import getters
from collections import Counter
generic = lambda x: ast.literal_eval(x)
conv = {'indexes': generic}

#%%
one = pd.read_csv("data_cleaned/pl/noexp/0.1/1819/[4, 3, 3].csv", converters = conv)
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

def is_diverse_ed2(playersdata, team_ids, s = 1/2, z_count = 4): # s determines interval length
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
    theory_int_l = [x - math.ceil(c/s) for x in theory_int]
    theory_int_h = [x + math.ceil(c/s) for x in theory_int]
    counts = [0]*11
#    print(c)
#    print(theory_int_l)
#    print(theory_int_h)
#    save_empty_indices = []
    print(team_cost)
    print(theory_int_l)
    print(theory_int_h)    
    
    for re in team_cost:
        
        for i,(low,up) in enumerate(zip(theory_int_l, theory_int_h)):
            if(re >= low and re <= up):
                counts[i] += 1
    print(counts)

    
    if(counts.count(0) < z_count): # zeros implicates amount that interval doesn't cover
        print(1)
        return(0)
    else:
        print(0)
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
from collections import Counter
#inter = thry_interval(a, 700)
#players = pd.read_csv("data/pl_csv/players_incnew_1819.csv")
#playerspl = players.to_dict('index')
playerspldata = getters.get_players_feature_pl("data/pl_csv/players_noexp_0.1_", 1819)
#cost_list = calc.createCostList(playerspldata, False)
#%%

budget= 500
ones = filter_df(one, 0, budget)
ones.sort_values(by ="points_total", inplace = True, ascending = False)

ones = ones.sample(n = 50)
all_teams = ones["indexes"].to_list()
#all_teams_cost_list = [get.get_cost_team(cost_list, team_id) for team_id in all_teams]
#testIfLinear(all_teams_cost_list[0], budget)

ss = Counter(flatten(all_teams)).most_common()
#all_teams = random.sample(all_teams, 50)

#%%

is_dev_or_not = [is_diverse_ed2(playerspldata, team_id, s = 1) for team_id in all_teams]

#print(sum([is_dev_or_not[x][0] for x in range(len(is_dev_or_not))]))
print(sum(is_dev_or_not))
#enum = [[i for (i,x) in enumerate(is_dev_or_not[y][1]) if x == 0]  for y in range(len(is_dev_or_not))]
#cunt = Counter(flatten(enum)).most_common()
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

##JONTE TESTAR
import scipy.stats as stats      
import matplotlib.pyplot as plt

def testNormalInter(h, plot=False):       
    mu = np.mean(h)
    sigma = np.std(h)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    
    high = mu+3*sigma
    low= mu+- 3*sigma
    
    low1 = mu-sigma
    high1 = mu+sigma
    tot=0
    for p in h:
        if low1 < p < high1:
            tot+=1
        perc1 = int(tot/len(h)*100)
        if perc1 > 68:
            norm1 = 'Normal' #should be higher than 68%
            ret=1
        else:
            norm1='Not normal'
            ret=0
    if plot:    
        plt.plot(x, stats.norm.pdf(x, mu, sigma)*2)
        plt.axvline(x=low1, color='r', ls='--')
        plt.axvline(x=high1, color='r', ls='--')
    #        # fit = stats.norm.pdf(h, np.mean(h), np.std(h))*2  #this is a fitting indeed
    #         #plt.plot(h,fit,'-o')
            
    #         #plt.axvline(x=mu+2*sigma, color='b', ls='--') # *1.645 for 90%
    #         #plt.axvline(x=mu-2*sigma, color='b', ls='--') # *1.960 for 95 %
            
    #         #plt.hist(h,nrinter, density = True)
        plt.hist(h,len(h),density=True)
        plt.title(norm1)
        plt.show()    
    return perc1, ret
     
def linR2Inter(h, ax, plot=False):
    
    X=range(len(h))
    Y= h
    
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    m = len(X)
    
    numer = 0
    denom = 0
    for i in range(m):
      numer += (X[i] - mean_x) * (Y[i] - mean_y)
      denom += (X[i] - mean_x) ** 2
    m = numer / denom
    c = mean_y - (m * mean_x)
    
    x = range(len(h))
    y = c + m * x
            
    ss_t = 0 #total sum of squares
    ss_r = 0 #total sum of square of residuals
    
    for i in range(len(h)): # val_count represents the no.of input x values
      ss_t += (Y[i] - mean_y) ** 2
      ss_r += (Y[i] - y[i]) ** 2
    r2 = 1 - (ss_r/ss_t)
    
    
    if r2> 0.9:
        ret = 0
        norm = 'Not normal'
    else:
        ret=1
        norm = 'Normal'
    if plot:
         ax.plot(x, y, color='r', label='Regression Line')
         ax.scatter(X, Y, c='b', label='Data points')
         ax.set_title(norm)
         ax.legend()    
    return r2, ret

def checkdiversity(playersdata, team_ids, ax=None, plot=False) : 
    h = get.get_cost_team(playersdata, team_ids)

    _, ret1 = linR2Inter(h, ax, plot)
    _, ret2 = testNormalInter(h,plot)
    
    if ret1==ret2:
        return ret1
    
    else:
        return -1

def printpercent(title, diverselist, nperc, divperc, undefperc):
    print(title + "nr normal:", diverselist.count(1), "percent:",nperc)
    print(title + "nr diverse:", diverselist.count(0), "percent:", divperc)
    print(title + "nr undefined:", diverselist.count(-1), "percent:", undefperc)

def calcpercent(divlist):
    nor = round(100*divlist.count(1)/len(divlist))
    div = round(100*divlist.count(0)/len(divlist))
    und = round(100*divlist.count(-1)/len(divlist))   
    
    return nor, div, und 

#%%
season= 1617
formation= '[3, 4, 3]'
one = pd.read_csv('data_cleaned/pl/'+str(season)+'/'+str(formation)+ '.csv', converters =conv)

#%%
#Create df for saving results 

dfres = pd.DataFrame(columns=['Budget interval', 'Best 50', 'Worst 50', 'All'])

startlow =450
endlow =1000
idx=0
for low in range(startlow, endlow,50):
    
    budget = low+50
    print('-------------------------------------')
    print(budget)
    ones = filter_df(one, low, budget)
    ones.sort_values(by ="points_total", inplace = True, ascending = False)
    playerspldata = get.get_players_feature_pl("data/pl_csv/players_raw_", 1617)
    all_teams = ones["indexes"].to_list()
    ss = Counter(flatten(all_teams)).most_common()
    allpoints= ones['points_total'].to_list() 
    
    #Take 50 best 
    best_50 = [ones.iloc[i]['indexes'] for i in range(50)]
    
    best_div=[]
    i=0
    plot= False
    for team in best_50:
        #for plotting
        #if plot==True:
        #    i+=1
        #    fig, (ax1, ax2) = plt.subplots(1, 2)
        #    fig.suptitle(i)
        # then add ax = ax1 in next row
        best_div.append(checkdiversity(playerspldata,team, plot))
        
    bnor, bdiv, bund = calcpercent(best_div)
    b50  = [bnor,bdiv,bund]
    
    #For printing
    #printpercent('Best 50', best_div, bnor, bdiv, bund)
    
    #Take 50 worst 
    w_50 = [ones.iloc[-i]['indexes'] for i in range(50)]
    w_div=[]
    
    for team in w_50:
        w_div.append(checkdiversity(playerspldata,team))
    
    wnormperc =round(100*w_div.count(1)/len(w_div))
    wdivperc= round(100*w_div.count(0)/len(w_div))
    wundefperc=round(100*w_div.count(-1)/len(w_div))  
    w50 = [wnormperc, wdivperc, wundefperc]
    

    #print("Worst 50 nr normal:", w_div.count(1), "percent:",wnormperc)
    #print("Worst 50 nr diverse:", w_div.count(0), "percent:", wdivperc)
    #print("Worst 50 nr undefined:", w_div.count(-1), "percent:", wundefperc)
    
    diverse=[]
    
    for team_id in all_teams:
        #i+=1
        #fig, (ax1, ax2) = plt.subplots(1, 2)
        #fig.suptitle(i)
        diverse.append(checkdiversity(playerspldata, team_id))
    
    anorm =round(100*diverse.count(1)/len(diverse))
    adiv=round(100*diverse.count(0)/len(diverse))
    aundef = round(100*diverse.count(-1)/len(diverse))
    #print("Nr normal:", diverse.count(1), "percent:", round(100*diverse.count(1)/len(diverse)))
    #print("Nr diverse:", diverse.count(0), "percent:", round(100*diverse.count(0)/len(diverse)))
    #print("Nr undefined:", diverse.count(-1), "percent:", round(100*diverse.count(-1)/len(diverse)))
    
    a=[anorm, adiv, aundef]

    dfres.loc[idx]=[str(low) + ' to ' + str(budget), b50,w50,a]
    idx+=1


dfres.to_csv('results/pl/'+ str(season) +'/perc_' +str(formation)+ '.csv')

#%%   
indexes_div = [i for (i,x) in enumerate(best_div) if x==0]
tot_points = []
tot_cost = []
for ind in indexes_div:
    tot_points.append(ones.iloc[ind]['points_total'])
    tot_cost.append(ones.iloc[ind]['cost'])
    
print((sum(tot_cost)/len(tot_cost)/ ones['cost'].mean()))
print((sum(tot_points)/len(tot_points))/ ones['points_total'].mean())    


#%%
print("Nr normal:", diverse.count(1), "percent:", round(100*diverse.count(1)/len(diverse)))
print("Nr diverse:", diverse.count(0), "percent:", round(100*diverse.count(0)/len(diverse)))
print("Nr undefined:", diverse.count(-1), "percent:", round(100*diverse.count(-1)/len(diverse)))


