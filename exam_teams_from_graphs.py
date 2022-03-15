# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:41:32 2022

@author: jonat
"""

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
import scipy.stats as stats      
import matplotlib.pyplot as plt
generic = lambda x: ast.literal_eval(x)
conv = {'indexes': generic}

#%%%
def filter_df(df, lwr, upper):
    df = df[df['cost'] <= upper]
    df_new = df[df['cost'] >= lwr]
    return(df_new)

def flatten_all_ids(all_teams_id):
    return(set([item for sublist in all_teams_id for item in sublist]))

def det_a(data, all_ids):
    return(min(get.get_cost_team(data,all_ids)))


def flatten(l):
    flattened = []
    for sublist in l:
        flattened.extend(sublist)
    return flattened

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

def testNormalInter(h, plot=False, intervals= 1):       
    mu = np.mean(h)
    sigma = np.std(h)
    
    low1 = mu-sigma
    high1 = mu+sigma
    tot1=0
    for p in h:
        if low1 < p < high1:
            tot1+=1
    perc1 = round(tot1/len(h)*100)
        
        
    if intervals == 1:
        if perc1 > 68:
            #       norm1 = 'Normal' #should be higher than 68%
            ret=1
        else:
    #        norm1='Not normal'
            ret=0
    elif intervals==2:
        low2= mu - 1.645*sigma
        high2= mu + 1.645*sigma
        tot2=0
        for p in h:
            if low2 < p < high2:
                tot2+=1
        perc2 = round(tot2/len(h)*100)
        
        if perc1 > 68 and perc2 > 90:
            norm1 = 'Normal' #should be higher than 68%
            ret=1
        else:
            norm1='Not normal'
            ret=0
             
    if plot: 
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        
        #high = mu+3*sigma
        #low= mu+- 3*sigma
        plt.plot(x, stats.norm.pdf(x, mu, sigma)*200)
        plt.axvline(x=low1, color='r', ls='--')
        plt.axvline(x=high1, color='r', ls='--')
            # fit = stats.norm.pdf(h, np.mean(h), np.std(h))*2  #this is a fitting indeed
            #plt.plot(h,fit,'-o')
            
        plt.axvline(x=mu+1.645*sigma, color='b', ls='--') # *1.645 for 90%
        plt.axvline(x=mu-1.645*sigma, color='b', ls='--') # *1.960 for 95 %
            
        #plt.hist(h,len(h), density = True)
        plt.hist(h,len(h))
        plt.title(norm1)
        plt.show()
    print(perc1, perc2)    
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
    print(r2)
    return r2, ret

def checkdiversity(playersdata, team_ids, ax=None,  plot=False) : 
    h = get.get_cost_team(playersdata, team_ids)
    print(h)

    r2, ret1 = linR2Inter(h, ax, plot)
    perc1, ret2 = testNormalInter(h,plot, intervals=2)
    
    if ret1==ret2:
        return ret1
    
    else:
        if r2> 93 or r2 < 87: 
            return ret1
        return -1

def printpercent(title, diverselist, nperc, divperc, undefperc):
    print(title + "nr normal:", diverselist.count(1), "percent:",nperc)
    print(title + "nr diverse:", diverselist.count(0), "percent:", divperc)
    print(title + "nr undefined:", diverselist.count(-1), "percent:", undefperc)

def calcpercent(divlist):
    if len(divlist)>0:
        nor = round(100*divlist.count(1)/len(divlist))
        div = round(100*divlist.count(0)/len(divlist))
        und = round(100*divlist.count(-1)/len(divlist))   
    
        return nor, div, und 
    else:
        return 0,0,0

#%%
#Create df for saving results 
seasons= [1718]
#formations= ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]','[4, 4, 2]','[4, 5, 1]','[5, 3, 2]', '[5, 4, 1]']
formations=['[3, 4, 3]']
for season in seasons:
    for formation in formations: 
        print('Preparing data', str(formation))
        one = pd.read_csv('data_cleaned/pl/'+str(season)+'/'+str(formation)+ '.csv', converters =conv)
        print('Done')
    
        useall = False
        if useall:
            dfres = pd.DataFrame(columns=['Budget interval', 'Best 50', 'All'])
        else: 
            dfres = pd.DataFrame(columns=['Budget interval', 'Best 50'])
        
        #startlow =450
        #endlow =1000
        startlow= 600
        endlow=650
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
            if len(ones)>50:
        
                best_50 = [ones.iloc[i]['indexes'] for i in range(50)]
            else:
                best_50 = [ones.iloc[i]['indexes'] for i in range(len(ones))]
            
            best_div=[]
            i=0
            plot= True
            for team in best_50:
                #for plotting
                #print(team)
                if plot==True:
                    i+=1
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                    fig.suptitle(i)
                #then add ax = ax1 in next row
                best_div.append(checkdiversity(playerspldata,team, ax1, plot))
                
            bnor, bdiv, bund = calcpercent(best_div)
            b50  = [bnor,bdiv,bund]
            
            #For printing
            #printpercent('Best 50', best_div, bnor, bdiv, bund)
            
            if useall: 
                diverse=[]
                for team_id in all_teams:
                    diverse.append(checkdiversity(playerspldata, team_id))
            
                anor, adiv, aund = calcpercent(diverse)
                a  = [anor,adiv,aund]
            
                #For printing
                #printpercent('All', diverse, anor, adiv, aund)
        
                dfres.loc[idx]=[str(low) + ' to ' + str(budget), b50,a]
            else: 
                dfres.loc[idx]=[str(low) + ' to ' + str(budget), b50]
            idx+=1
        
        
        #dfres.to_csv('results/pl/'+ str(season) +'/perc_' +str(formation)+ '.csv') 
