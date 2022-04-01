# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:42:40 2022

@author: jgull
"""
import pandas as pd
#import cleaners
import clss
import ast
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import csv
import getters as get
import numpy as np
generic = lambda x: ast.literal_eval(x)
conv = {'indexes': generic}
#%%

def exam_r2_plots(budg):
    empt = 5*[True]
    i=0
    for team in budg.team_list:
        print(team.r2)
        #colors = cm.rainbow(np.linspace(0, 1, len(budg.team_list)+1))
                
        for r1, r2 in zip(range(5,10), range(6,11)):
            print(r1)
            if(team.r2 > r1/10 and team.r2 <= r2/10 and empt[r1-5]):
                empt[r1-5] = False
                i+=1
                        
                fig, ax = plt.subplots()
                ax.scatter(range(11), team.cost, marker = 'o', facecolors= "none", edgecolors='r')
                plt.title(team.r2)
                plt.show()
                break
    
seasons = [1819]

typ = "raw"
formations= ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]','[4, 4, 2]','[4, 5, 1]','[5, 3, 2]']

#Produce diff r2 values:
for season in seasons:

    for formation in formations:
        print('Preparing data', str(formation))
        loc =  'data_cleaned/pl/' + str(season)+'/'
        one = pd.read_csv(loc+str(formation)+ '.csv', converters =conv)
        playerspldata = get.get_players_feature_pl("data/pl_csv/players_raw_", season)
        one.sort_values(by="points_total", inplace=True, ascending = False)

        print('Done')
    
        useall = True   # T -> alla, F -> bara 50 bästa
        if useall:
            dfres = pd.DataFrame(columns=['Budget interval', 'Best 50', 'Worst 50', 'All'])
        else:
            dfres = pd.DataFrame(columns=['Budget interval', 'Best 50', 'Worst 50'])
        startlow = 450
        endlow = 950
        idx = 0
        all_budg = {}
        for low in range(startlow, endlow, 50):
            
            budget = low+50
            
            print('-------------------------------------')
            print(budget)
            ones = cleaners.filter_df(one, low, budget)
            all_teams = ones["indexes"].to_list()
            allpoints = ones['points_total'].to_list()
    
                    #Take 50 best
            if len(ones) > 50:
    
                best_50 = ones.nlargest(50, "points_total")
            else:
                best_50 = ones.nlargest(len(ones),"points_total")
            inds = best_50["indexes"].tolist()
            points = best_50["points_total"].tolist()
            
            rest = np.array([])
            i = 0
            plot = False
            linst = []
            budg = clss.budget(low, budget, season, typ)
            budg.set_teams(inds, points, playerspldata)
            budg.lin_reg()
    #        budg.get_all_res()
   #         fig, ax = plt.subplots()
            dest = os.path.join(loc, str(budget))
            #os.makedirs(dest, exist_ok = True)
            temp = {}
            
            for r2 in range(5,10):
                r2 = r2/10
                budg.get_is_linear(r2)
                temp[r2] = budg.prop_lin
            all_budg[budget] = temp
        dataf = pd.DataFrame.from_dict(all_budg, orient = "index") 
        i=0
        dataf.to_csv("results/pl/" + str(season) + "/r2/" + str(formation) +"_n" ".csv")
            #exam_r2_plots(budg)
           # if len(budg.res)>0:
           #     
         #       plt.savefig("res.png")
     
#%%
# investigate all:
#for season in seasons:
formations= ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]','[4, 4, 2]','[4, 5, 1]','[5, 3, 2]']
ses = {}
typ="raw"
seasons = [1718,1819,1920,2021]
import time
from sklearn.linear_model import LinearRegression as lin
import getters
import numpy as np
class fast_lin_reg:
    def __init__(self):
        self = self
    def fit(self, x,y, r2lim):
        mean_x = np.mean(x)
        mean_y = np.mean(y)

        sub_mean_x = np.subtract(x, mean_x)
        sub_mean_y = np.subtract(y,mean_y)

        prod = np.multiply(sub_mean_x,sub_mean_y)

        numer = np.sum(prod)
        denom = np.sum(np.power(sub_mean_x,2))

        m = np.divide(numer,denom)
        c = np.subtract(mean_y,np.multiply(m,mean_x))

        y_pr = np.add(c,np.multiply(m,x))
        ss_t = np.sum(np.power((y - mean_y),2))
        ss_r = np.sum(np.power((y - y_pr), 2))         

        self.r2 = np.subtract(1,np.divide(ss_r,ss_t)) 

        if self.r2 >= r2lim: self.linear = True
        else: self.linear = False
#%%
seasons = [1617, 1718,1819,1920,2021]
seasons= [1617]
formations= ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]','[4, 4, 2]','[4, 5, 1]','[5, 3, 2]','[5, 4, 1]']
typ = "raw"
for season in seasons:
    print(str(season))
    lin_cost, lin_points, lin_n, nonlin_cost, nonlin_points, nonlin_n = [],[],[],[],[],[]
    for formation in formations:
        print('Preparing data', str(formation))
        if typ == "raw":
            loc =  'data_cleaned/pl/'+str(season)+'/'
            one = pd.read_csv(loc+str(formation)+ '.csv', converters =conv)
        elif typ == "incnew": 
            loc = 'data_cleaned/pl/'+ typ + "/"
            one = pd.read_csv(loc +str(season)+'/'+str(formation)+ '.csv', converters =conv)
        elif typ == "noexp":
            loc = 'data_cleaned/pl/'+ "noexp_01" + "/"
            one = pd.read_csv(loc +str(season)+'/'+str(formation)+ '.csv', converters =conv)

        playerspldata = get.get_players_feature_pl("data/pl_csv/players_" + typ + "_", season)
        one.sort_values(by="points_total", inplace=True, ascending = False)
    
        print('Done')
       
    #       useall =    # T -> alla, F -> bara 50 bästa
     #      if useall:
      #         dfres = pd.DataFrame(columns=['Budget interval', 'Best 50', 'Worst 50', 'All'])
       #    else:
       #        dfres = pd.DataFrame(columns=['Budget interval', 'Best 50', 'Worst 50'])
        all_teams = one["indexes"].to_list()
        all_points = one['points_total'].to_list()
        all_costs = one['cost'].to_list()
       
        linear_cost, linear_points, non_cost, non_points = [],[],[],[]
        x = np.linspace(0,10,11).reshape(-1)
        j, sums= 0,0
        for t,p,c in zip(all_teams, all_points, all_costs):
            y = get.get_cost_team(playerspldata, t)
            linmod = fast_lin_reg()
            linmod.fit(x,y, 0.85)
            if(linmod.linear):
                linear_cost.append(c)
                linear_points.append(p)
            else:
                non_cost.append(c)
                non_points.append(p)
        break
    lin_cost.append(np.mean(linear_cost))
    lin_points.append(np.mean(linear_points))
    lin_n.append(len(linear_cost))
        
    nonlin_cost.append(np.mean(non_cost))
    nonlin_points.append(np.mean(non_points))
    nonlin_n.append(len(non_cost))
    df = pd.DataFrame({'Linear mean cost': lin_cost, 'Linear mean points': lin_points, 'n linear': lin_n,
                  'Non linear cost': nonlin_cost, 'Non linear points': nonlin_points, 'Non linear n': nonlin_n
        })
    
    dest = os.path.join("results","pl",str(season), "linvsnonlin_" + typ + ".csv")
    df.to_csv(dest)

