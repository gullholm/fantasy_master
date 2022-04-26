#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:38:11 2022

@author: josef
"""
#%%%

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
import parsers
import helpers_calc_div
import math
import scipy.stats as stats      
import matplotlib.pyplot as plt



#%%
one = pd.read_csv("data_cleaned/pl/1718/[3, 4, 3].csv", converters = conv)
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
    inter = np.linspace(a+c, (2*upper/11)-a, 11, dtype = float)
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

def is_diverse_ed2(playersdata, team_ids, s = 2, z_count = 4, rang = 2): # s determines interval length
    team_cost = get.get_cost_team(playersdata, team_ids)
#    print(team_id)
    team_cost.sort()
#    print(team_cost)
#    a = det_a(full_data, team_id

    a = min(team_cost)
    b= max(team_cost)
#    print(team_cost)
    theory_int = np.round(np.linspace(a, b, 11, dtype=float),1).tolist()
    
#    print(theory_int)
    c = np.round((theory_int[1]-theory_int[0])/2,1)
    theory_int_l = [round(x - c,1) for x in theory_int]
    theory_int_h = [round(x + c,2) for x in theory_int]
    counts = [0]*11
#    print(c)
#    print(theory_int_l)
#    print(theory_int_h)
#    save_empty_indices = []
#    print(team_cost)
#    print(theory_int)
#    print(theory_int_h)    
#    
    for re in team_cost:
        
        for i,(low,up) in enumerate(zip(theory_int_l, theory_int_h)):
            if(re >= low and re <= up):
                counts[i] += 1
                break
#    print(counts)
    mas = max(counts)
    max_ind = [i for (i,x) in enumerate(counts) if x == mas]
    normals = []
    for i in max_ind:
#        print(i)
        if i == 0: i+=rang
#        if i == 1: i +=1
        if i == 10: i-=rang
#        if i == 9: i-=1
        normals.append(sum([counts[j] for j in range(i-rang,i+rang+1)]))
#    print(normals)
    if(sum(counts)> 11): print("hi")    
    normal = max(normals)
#    print(counts)
#    print(normal)
    if(normal > z_count): 
#        print("normal")        
        return(0)
    else:
#        print(counts)
#        print("linear")
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
    

##JONTE TESTAR


def testNormalInter(h, plot=False, intervals= 1):       
    mu = np.mean(h)
    sigma = np.std(h)
    #x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    
    #high = mu+3*sigma
    #low= mu+- 3*sigma
    
    low1 = mu-sigma
    high1 = mu+sigma
    tot1=0
    for p in h:
        if low1 < p < high1:
            tot1+=1
    perc1 = int(tot1/len(h)*100)
        
        
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
        tot1=0
        tot2=0
        for p in h:
            if low2 < p < high2:
                tot2+=1
        perc2 = int(tot2/len(h)*100)
        
        if perc1 > 68 and perc2 > 90:
            #       norm1 = 'Normal' #should be higher than 68%
            ret=1
        else:
    #        norm1='Not normal'
            ret=0
             
    #if plot:    
    #    plt.plot(x, stats.norm.pdf(x, mu, sigma)*2)
    #    plt.axvline(x=low1, color='r', ls='--')
    #    plt.axvline(x=high1, color='r', ls='--')
    #        # fit = stats.norm.pdf(h, np.mean(h), np.std(h))*2  #this is a fitting indeed
    #         #plt.plot(h,fit,'-o')
            
    #         #plt.axvline(x=mu+2*sigma, color='b', ls='--') # *1.645 for 90%
    #         #plt.axvline(x=mu-2*sigma, color='b', ls='--') # *1.960 for 95 %
            
    #         #plt.hist(h,nrinter, density = True)
    #    plt.hist(h,len(h),density=True)
    #    plt.title(norm1)
    #    plt.show()    
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
    
    res = [i-I for (i,I) in zip(y,Y)]
    if r2> 0.85:
        ret = 0
    #    norm = 'Not normal'
    #    return r2, ret, res
    else:
        ret=1
    #    norm = 'Normal'
    #if plot:
    #     ax.plot(x, y, color='r', label='Regression Line')
    #     ax.scatter(X, Y, c='b', label='Data points')
    #     ax.set_title(norm)
    #     ax.legend()    
    return r2, ret, res

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
    if len(divlist)>0:
        nor = round(100*divlist.count(1)/len(divlist))
        
        div = round(100*divlist.count(0)/len(divlist))
        #und = round(100*divlist.count(-1)/len(divlist))   
    
        return nor, div#, und 
    else:
        return 0,0#,0
     

#%%%
#all_ids = list(flatten_all_ids(one["indexes"].to_list()))
#a = det_a(data, all_ids)


#inter = thry_interval(a, 700)
#players = pd.read_csv("data/pl_csv/players_incnew_1819.csv")
#playerspl = players.to_dict('index')
playerspldata = getters.get_players_feature_pl("data/pl_csv/players_raw_", 1718)
#cost_list = calc.createCostList(playerspldata, False)
ones= filter_df(one, 650, 700)
all_teams = ones["indexes"].to_list()
random.seed(123)
#all_teams = random.sample(all_teams, 50)
isd = [is_diverse_ed2(playerspldata, team_id, s = 2, z_count = 7, rang =1) for team_id in all_teams]
indexes_div = [i for (i,x) in enumerate(isd) if x==1]
tot_points = []
tot_cost = []
#all_points = ones.total_points.tolist()
#all_costs = ones.cost.tolist()

#tot_points = [all_points[i] for i in indexes_div]
for ind in indexes_div:
     tot_points.append(ones.iloc[ind]['points_total'])
     tot_cost.append(ones.iloc[ind]['cost'])


print((sum(tot_cost)/len(tot_cost)/ ones['cost'].mean()))
print((sum(tot_points)/len(tot_points))/ ones['points_total'].mean())
#%%


budget= [550,600,650,700,750,800,850]

for budg in budget:
    ones = filter_df(one, 0, budg)
    ones.sort_values(by ="points_total", inplace = True, ascending = False)


#ones = ones.sample(n = 50)
    all_teams = ones["indexes"].to_list()
#all_teams_cost_list = [get.get_cost_team(cost_list, team_id) for team_id in all_teams]
#testIfLinear(all_teams_cost_list[0], budget)

    ss = Counter(flatten(all_teams)).most_common()
#all_teams = random.sample(all_teams, 50)

    is_dev_or_not = [is_diverse_ed2(playerspldata, team_id, s = 1) for team_id in all_teams]

#print(sum([is_dev_or_not[x][0] for x in range(len(is_dev_or_not))]))
    print(sum(is_dev_or_not))
#enum = [[i for (i,x) in enumerate(is_dev_or_not[y][1]) if x == 0]  for y in range(len(is_dev_or_not))]
#cunt = Counter(flatten(enum)).most_common()

#print(sum(is_dev_or_not))
    indexes_div = [i for (i,x) in enumerate(is_dev_or_not) if x==1]
    tot_points = []
    tot_cost = []
    for ind in indexes_div:
        tot_points.append(ones.iloc[ind]['points_total'])
        tot_cost.append(ones.iloc[ind]['cost'])
    print(budget)
    print((sum(tot_cost)/len(tot_cost)/ ones['cost'].mean()))
    print((sum(tot_points)/len(tot_points))/ ones['points_total'].mean())



#%%
seasons= [1617,1718,1819,1920,2021]
formations= ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]','[4, 4, 2]','[4, 5, 1]','[5, 3, 2]', '[5, 4, 1]']
season= seasons[0]
formation= formations[0]
one = pd.read_csv('data_cleaned/pl/'+str(season)+'/'+str(formation)+ '.csv', converters =conv)

#%% josef testar

c_list = []
for team in all_teams:
    cc = get.get_cost_team(playerspldata, team)
    if (cc[1] == 1):
        c_list.extend(get.get_cost_team(playerspldata, team)[2])


#%%
#Create df for saving results 
seasons= [1718]
#Formations without 343 for the monment since it is already done
formations= ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]','[4, 4, 2]','[4, 5, 1]','[5, 3, 2]', '[5, 4, 1]']

for season in seasons:
    for formation in formations: 
        print('Preparing data', str(formation))
        one = pd.read_csv('data_cleaned/pl/'+str(season)+'/'+str(formation)+ '.csv', converters =conv)
        print('Done')
    
        useall = True   # T -> alla, F -> bara 50 bästa
        if useall:
            dfres = pd.DataFrame(columns=['Budget interval', 'Best 50', 'Worst 50', 'All'])
        else: 
            dfres = pd.DataFrame(columns=['Budget interval', 'Best 50', 'Worst 50'])
        
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
            #ss = Counter(flatten(all_teams)).most_common()
            #allpoints= ones['points_total'].to_list() 
            
            #Take 50 best 
            if len(ones)>50:
        
                best_50 = [ones.iloc[i]['indexes'] for i in range(50)]
            else:
                best_50 = [ones.iloc[i]['indexes'] for i in range(len(ones))]
            
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
            if len(ones)>50:
        
                w_50 = [ones.iloc[-i]['indexes'] for i in range(50)]       
            else:
                w_50 = [ones.iloc[-i]['indexes'] for i in range(len(ones))]
            
            w_div=[]
            
            for team in w_50:
                w_div.append(checkdiversity(playerspldata,team))
            
            wnor, wdiv, wund = calcpercent(w_div)
            w50  = [wnor,wdiv,wund]
            
            #For printing
            #printpercent('Worst 50', w_div, wnor, wdiv, wund)
            
            if useall: 
                diverse=[]
                for team_id in all_teams:
                    diverse.append(checkdiversity(playerspldata, team_id))
            
                anor, adiv, aund = calcpercent(diverse)
                a  = [anor,adiv,aund]
            
                #For printing
                #printpercent('All', diverse, anor, adiv, aund)
        
                dfres.loc[idx]=[str(low) + ' to ' + str(budget), b50,w50,a]
            else: 
                dfres.loc[idx]=[str(low) + ' to ' + str(budget), b50,w50]
            idx+=1
        
        
        dfres.to_csv('results/pl/'+ str(season) +'/perc_' +str(formation)+ '.csv') 



#%%

#create formations
getformations = parsers.write_full_teams('data_cleaned/pl/2021/')

#%%
dfRes = pd.DataFrame()
seasons=[1617,1718,1819,1920,2021]

allworstlist =[[0,0,0]]*11
allbestlist = [[0,0,0]]*11

for season in seasons: 
    
    meanworstlist= []
    meanbestlist = []
    
    for idx in range(11):
        print(idx)
        formations= ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]','[4, 4, 2]','[4, 5, 1]','[5, 3, 2]']
        allworst = []
        allbest = []
        
        for formation in formations:
        
            #idx=3
            res = pd.read_csv('results/pl/' +str(season) + '/perc_' + str(formation) + '.csv')
            wor = ast.literal_eval(res['Worst 50'][idx])
            bes = ast.literal_eval(res['Best 50'][idx])
            al =  ast.literal_eval(res['All'][idx])
            
            worratio = [i/j if j!=0 else None for i,j in zip(wor, al)]
            besratio = [i/j if j!=0 else None for i,j in zip(bes, al)] 
            
            allworst.append(worratio)
            allbest.append(besratio)
        
        #print(allbest)
        sumbest =[-1]*3
        sumworst = [-1]*3
        for i in range(3):    
            sumbest[i] = [j[i] for j in allbest if j[i] != None]
            sumworst[i] = [j[i] for j in allworst if j[i] != None]
            
        #print(sumbest)    
        meanbest = [sum(k)/len(sumbest) for k in sumbest]  
        meanworst = [sum(k)/len(sumworst) for k in sumworst]    
        
        meanworstlist.append(meanworst)
        meanbestlist.append(meanbest)
    
    
    for i,j in enumerate(meanworstlist):

        allworstlist[i] = [(a+b)/5 for a,b in zip(j,allworstlist[i])]
        allbestlist[i] = [(c+d)/5 for c,d in zip(j,allbestlist[i])]
    
    dfRes[str(season) + ' worst'] = meanworstlist
    dfRes[str(season) + ' best'] = meanbestlist


dfRes.to_csv('results/pl/ratio_normal')    


#%%

#Calc ratios 


#Plot piechart of results    
formations= ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]','[4, 4, 2]','[4, 5, 1]','[5, 3, 2]', '[5, 4, 1]']


for formation in formations:

    res = pd.read_csv('results/pl/1617/perc_' + str(formation) + '.csv')
    for i in range(11):
        resbest = ast.literal_eval(res['Best 50'][i])
        resworst = ast.literal_eval(res['Worst 50'][i]) 
        resall = ast.literal_eval(res['All'][i])

#%%
#PL
#Create df for saving results, checking linearity of all teams, linear or not according to R2
 
seasons= [1617,1718,1819,1920,2021]
#seasons= [1920, 2021]
seasons=[2021]
formations= ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]','[4, 4, 2]','[4, 5, 1]','[5, 3, 2]', '[5, 4, 1]']
formations=['[3, 5, 2]']
for season in seasons:
    print(season)
    for formation in formations: 
        print('Preparing data', str(formation))
        one = pd.read_csv('data_cleaned/pl/'+str(season)+'/'+str(formation)+ '.csv', converters =conv)
        print('Done')
    
        useall = True  # T -> alla, F -> bara 50 bästa
        if useall:
            dfres = pd.DataFrame(columns=['Budget interval', 'Best 50 (Normal,Diverse)', 'All (Normal,Diverse)'])
        else: 
            dfres = pd.DataFrame(columns=['Budget interval', 'Best 50 (Normal, Diverse)'])
        
        startlow =450
        endlow =1000
        idx=0
        for low in range(startlow, endlow,50):
            
            budget = low+50
            print('-------------------------------------')
            print(budget)
            ones = filter_df(one, low, budget)
            ones.sort_values(by ="points_total", inplace = True, ascending = False)
            playerspldata = get.get_players_feature_pl("data/pl_csv/players_raw_", season)
            all_teams = ones["indexes"].to_list()
            #ss = Counter(flatten(all_teams)).most_common()
           # allpoints= ones['points_total'].to_list() 
            
            #Take 50 best 
            if len(ones)>50:
                best_50 = [ones.iloc[i]['indexes'] for i in range(50)]
            else:
                best_50 = [ones.iloc[i]['indexes'] for i in range(len(ones))]
            
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
                h = get.get_cost_team(playerspldata, team)
                
                _ , ret, _ = linR2Inter(h, None, plot)
                #print(r2)
                best_div.append(ret)
                #print(best_div)
            bnor, bdiv = calcpercent(best_div)
            b50  = [bnor,bdiv]
            
            #For printing
            #printpercent('Best 50', best_div, bnor, bdiv, bund)
            
            if useall: 
                diverse=[]
                for team_id in all_teams:
                    h = get.get_cost_team(playerspldata, team_id)
                    _, ret, _ = linR2Inter(h, None, plot)
                    diverse.append(ret)
                
                anor, adiv = calcpercent(diverse)
                a  = [anor,adiv]
            
                #For printing
                #printpercent('All', diverse, anor, adiv, aund)
        
                dfres.loc[idx]=[str(low) + ' to ' + str(budget), b50,a]
            else: 
                dfres.loc[idx]=[str(low) + ' to ' + str(budget), b50]
            idx+=1
        
        
        dfres.to_csv('results/pl/'+ str(season) +'/linperc_' +str(formation)+ '.csv')     
        
#%%
#Kör Josefs checkinterval
#Create df for saving results, checking linearity of all teams, linear or not according to R2

seasons= [1617,1718,1819,1920,2021]
zvalue=3
formations= ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]','[4, 4, 2]','[4, 5, 1]','[5, 3, 2]', '[5, 4, 1]']
useall = True  # T -> alla, F -> bara 50 bästa

for season in seasons:
    print(season)
    for formation in formations: 
        print('Preparing data', str(formation))
        one = pd.read_csv('data_cleaned/pl/'+str(season)+'/'+str(formation)+ '.csv', converters =conv)
        print('Done')
    
        if useall:
            dfres = pd.DataFrame(columns=['Budget interval', 'Best 50 (Normal,Diverse)', 'All (Normal,Diverse)'])
        else: 
            dfres = pd.DataFrame(columns=['Budget interval', 'Best 50 (Normal, Diverse)'])
        
        startlow =450
        endlow =1000
        idx=0
        for low in range(startlow, endlow,50):
            
            budget = low+50
            print('-------------------------------------')
            print(budget)
            ones = filter_df(one, low, budget)
            ones.sort_values(by ="points_total", inplace = True, ascending = False)
            playerspldata = get.get_players_feature_pl("data/pl_csv/players_raw_", season)
            all_teams = ones["indexes"].to_list()
            #ss = Counter(flatten(all_teams)).most_common()
            #allpoints= ones['points_total'].to_list() 
            
            #Take 50 best 
            if len(ones)>50:
                best_50 = [ones.iloc[i]['indexes'] for i in range(50)]
            else:
                best_50 = [ones.iloc[i]['indexes'] for i in range(len(ones))]
            
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
                #h = get.get_cost_team(playerspldata, team)
                
               # _ , ret, _ = linR2Inter(h, None, plot)
                #print(r2)
                #print(team)
                each_team = helpers_calc_div.team(team,playerspldata)
                each_team.create_int()
                each_team.check_int()
                #print(each_team)
                #print(each_team.zero_count)
                
                if each_team.zero_count <=zvalue:
                    
                    best_div.append(0)
                else: 
                    best_div.append(1)
                #print(best_div)
            bnor, bdiv = calcpercent(best_div)
            b50  = [bnor,bdiv]
            
            #For printing
            #printpercent('Best 50', best_div, bnor, bdiv, bund)
            
            if useall: 
                diverse=[]
                print("Calc all teams")
                for team_id in all_teams:
            #        h = get.get_cost_team(playerspldata, team_id)
            #        _, ret, _ = linR2Inter(h, None, plot)
                    each_team = helpers_calc_div.team(team_id ,playerspldata)
                    each_team.create_int()
                    each_team.check_int()
                    #print(each_team)
                    #print(each_team.zero_count)
                    #print(each_team.zero_count)
                    
                    if each_team.zero_count <=zvalue:
                        
                        diverse.append(0)
                    else: 
                        diverse.append(1) 
                    #print(diverse)    
                anor, adiv = calcpercent(diverse)
                a  = [anor,adiv]
            
                #For printing
                #printpercent('All', diverse, anor, adiv, aund)
        
                dfres.loc[idx]=[str(low) + ' to ' + str(budget), b50,a]
            else: 
                dfres.loc[idx]=[str(low) + ' to ' + str(budget), b50]
            idx+=1
        
        
        dfres.to_csv('results/pl/'+ str(season) +'/intervalperc_' +str(formation)+ '.csv')

#%%
seasons = [1617, 1718, 1819, 1920, 2021]
formations= ['3-4-3', '3-5-2', '4-3-3', '4-4-2', '4-5-1', '5-3-2', '5-4-1']
dfres = pd.DataFrame(columns=[ 'Season', 'Formation', 'Budget', 'Best total cost', 'Best total points', 'Individual costs', 'Sorted individual costs', 'Id' ])
all_ids=[]
#formationdf = pd.Series([])
#seasondf =pd.Series([])
#i = 0
for season in seasons: 
#    for idx in range(77):
#        print(i)
#        print(season)
#        seasondf[i+idx]=season
#    i=i+77    
    for form in formations:
        teams = pd.read_csv('results/pl/'+str(season)+'/'+str(form)+ '.csv', converters =conv)
        dfres = pd.concat([dfres, teams])
        dfres.iloc[1]['Season']=1
        
        #id_list = teams['Id'].tolist()
        #all_ids.append(id_list)
        
#dfres.set_index([pd.Index([1,2,3]),])
#dfres.insert(0,"Season", seasondf)

dfres.to_csv('results/pl/best1Id.csv')

#%%
# Creating for the top 1 teams
#Create df for saving results, checking linearity of all teams, linear or not according to R2

print('Preparing data')
one = pd.read_csv('results/pl/best1Id.csv', converters =conv)
print('Done')

dfres = pd.DataFrame(columns=['Normal','Diverse'])
best_div=[]
for cost in one['Sorted individual costs']:
    h= ast.literal_eval(cost)

    _ , ret, _ = linR2Inter(h, None, plot=False)
    best_div.append(ret)
    
nor, div = calcpercent(best_div)
#nordiv  = [nor, div] 

dfres.loc[0]=[nor,div]    

#%%

print(dfres)
dfres.to_csv('results/pl/linperc_best1.csv') 

#%%
fig, ax = plt.subplots()
values= [best_div.count(1), best_div.count(0)]
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = (round(pct*total/100.0))
        return '{p:.0f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct
plt.pie(values,explode=[0.0,0.1], autopct=make_autopct(values) ,labels=['Not linear', 'Linear' ])
ax.set_title("Best teams for all budgets and seasons")
plt.show()

#%%
seasons= [1617,1718,1819, 1920,2021]
allforms=[]
for season in seasons: 
    
    print(str(season))
    data = pd.read_csv('results/pl/'+str(season)+'/best.csv', converters =conv)

    
    seasonform = [ast.literal_eval(d) for d in data['Formation']]
    allforms.extend(seasonform)

#%%
formations= [[3, 4, 3],[3, 5, 2],[4, 3, 3],[4, 4, 2],[4, 5, 1],[5, 3, 2],[5, 4, 1]]
formnames= ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]','[4, 4, 2]','[4, 5, 1]','[5, 3, 2]', '[5, 4, 1]']

formcounts=[]    
for form in formations:
    formcounts.append(allforms.count(form))
    
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

ax.bar(formnames,formcounts)
ax.set_ylabel('Count')
ax.set_xlabel('Formation')
ax.set_title('Best teams formations')
plt.show()

#%%
budgets=range(500,1050,50)
meanformsperbud=[]
for budget in budgets: 
    
    data = pd.read_csv('results/pl/budget/'+str(budget)+'.csv', converters =conv)

    seasonform = [ast.literal_eval(d) for d in data['Formation']]
    df =  [a[0] for a in seasonform]
    mf = [a[1] for a in seasonform]
    fw = [a[2] for a in seasonform]
    meandf = sum(df)/5  
    meanmf = sum(mf)/5
    meanfw = sum(fw)/5

    meanformsperbud.append([meandf, meanmf,meanfw])
    
plt.plot(range(500,1050,50), meanformsperbud, '-o')  
plt.title('Mean amount for each position in each budget')
plt.legend(['Defenders', 'Midfielders', 'Forwards'])  
plt.xlabel('Budget')
plt.ylabel('Amount')

#%%

getformations = parsers.write_full_teams('data_cleaned/pl/incnew/1819/')
        
#%%

#INCNEW
#Create df for saving results, checking linearity of all teams, linear or not according to R2
 
seasons= [1819]
formations= ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]','[4, 4, 2]','[4, 5, 1]','[5, 3, 2]', '[5, 4, 1]']

for season in seasons:
    print(season)
    for formation in formations: 
        print('Preparing data', str(formation))
        one = pd.read_csv('data_cleaned/pl/incnew/'+str(season)+'/'+str(formation)+ '.csv', converters =conv)
        print('Done')
    
        useall = True  # T -> alla, F -> bara 50 bästa
        if useall:
            dfres = pd.DataFrame(columns=['Budget interval', 'Best 50 (Normal,Diverse)', 'All (Normal,Diverse)'])
        else: 
            dfres = pd.DataFrame(columns=['Budget interval', 'Best 50 (Normal, Diverse)'])
        
        startlow =450
        endlow =1000
        idx=0
        for low in range(startlow, endlow,50):
            
            budget = low+50
            print('-------------------------------------')
            print(budget)
            ones = filter_df(one, low, budget)
            ones.sort_values(by ="points_total", inplace = True, ascending = False)
            playerspldata = get.get_players_feature_pl("data/pl_csv/players_raw_", season)
            all_teams = ones["indexes"].to_list()
            #ss = Counter(flatten(all_teams)).most_common()
           # allpoints= ones['points_total'].to_list() 
            
            #Take 50 best 
            if len(ones)>50:
                best_50 = [ones.iloc[i]['indexes'] for i in range(50)]
            else:
                best_50 = [ones.iloc[i]['indexes'] for i in range(len(ones))]
            
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
                h = get.get_cost_team(playerspldata, team)
                
                _ , ret, _ = linR2Inter(h, None, plot)
                #print(r2)
                best_div.append(ret)
                #print(best_div)
            bnor, bdiv = calcpercent(best_div)
            b50  = [bnor,bdiv]
            
            #For printing
            #printpercent('Best 50', best_div, bnor, bdiv, bund)
            
            if useall: 
                diverse=[]
                for team_id in all_teams:
                    h = get.get_cost_team(playerspldata, team_id)
                    _, ret, _ = linR2Inter(h, None, plot)
                    diverse.append(ret)
                
                anor, adiv = calcpercent(diverse)
                a  = [anor,adiv]
            
                #For printing
                #printpercent('All', diverse, anor, adiv, aund)
        
                dfres.loc[idx]=[str(low) + ' to ' + str(budget), b50,a]
            else: 
                dfres.loc[idx]=[str(low) + ' to ' + str(budget), b50]
            idx+=1
        
        
        dfres.to_csv('results/pl/'+ str(season) +'/incnew/linperc_' +str(formation)+ '.csv')     

#%%
#getformations = parsers.write_full_teams('data_cleaned/pl/noexp/1617/')
getformations = parsers.write_full_teams('data_cleaned/pl/noexp/1819/')

#%%

#NOEXP
#Create df for saving results, checking linearity of all teams, linear or not according to R2
 
seasons= [1617,1819]
formations= ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]','[4, 4, 2]','[4, 5, 1]','[5, 3, 2]', '[5, 4, 1]']

for season in seasons:
    print(season)
    for formation in formations: 
        print('Preparing data', str(formation))
        one = pd.read_csv('data_cleaned/pl/noexp/'+str(season)+'/'+str(formation)+ '.csv', converters =conv)
        print('Done')
    
        useall = True  # T -> alla, F -> bara 50 bästa
        if useall:
            dfres = pd.DataFrame(columns=['Budget interval', 'Best 50 (Normal,Diverse)', 'All (Normal,Diverse)'])
        else: 
            dfres = pd.DataFrame(columns=['Budget interval', 'Best 50 (Normal, Diverse)'])
        
        startlow =450
        endlow =1000
        idx=0
        for low in range(startlow, endlow,50):
            
            budget = low+50
            print('-------------------------------------')
            print(budget)
            ones = filter_df(one, low, budget)
            ones.sort_values(by ="points_total", inplace = True, ascending = False)
            playerspldata = get.get_players_feature_pl("data/pl_csv/players_raw_", season)
            all_teams = ones["indexes"].to_list()
            #ss = Counter(flatten(all_teams)).most_common()
           # allpoints= ones['points_total'].to_list() 
            
            #Take 50 best 
            if len(ones)>50:
                best_50 = [ones.iloc[i]['indexes'] for i in range(50)]
            else:
                best_50 = [ones.iloc[i]['indexes'] for i in range(len(ones))]
            
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
                h = get.get_cost_team(playerspldata, team)
                
                _ , ret, _ = linR2Inter(h, None, plot)
                #print(r2)
                best_div.append(ret)
                #print(best_div)
            bnor, bdiv = calcpercent(best_div)
            b50  = [bnor,bdiv]
            
            #For printing
            #printpercent('Best 50', best_div, bnor, bdiv, bund)
            
            if useall: 
                diverse=[]
                for team_id in all_teams:
                    h = get.get_cost_team(playerspldata, team_id)
                    _, ret, _ = linR2Inter(h, None, plot)
                    diverse.append(ret)
                
                anor, adiv = calcpercent(diverse)
                a  = [anor,adiv]
            
                #For printing
                #printpercent('All', diverse, anor, adiv, aund)
        
                dfres.loc[idx]=[str(low) + ' to ' + str(budget), b50,a]
            else: 
                dfres.loc[idx]=[str(low) + ' to ' + str(budget), b50]
            idx+=1
        
        
        dfres.to_csv('results/pl/'+ str(season) +'/noexp/linperc_' +str(formation)+ '.csv')     

#%%
import load_data
#AS
#Create df for saving results, checking linearity of all teams, linear or not according to R2
 

formations= ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]','[4, 4, 2]','[4, 5, 1]','[5, 3, 2]', '[5, 4, 1]']

for formation in formations: 
    print('Preparing data', str(formation))
    one = pd.read_csv('data_cleaned/as/'+str(formation)+ '.csv', converters =conv)
    print('Done')

    useall = True  # T -> alla, F -> bara 50 bästa
    if useall:
        dfres = pd.DataFrame(columns=['Budget interval', 'Best 50 (Normal,Diverse)', 'All (Normal,Diverse)'])
    else: 
        dfres = pd.DataFrame(columns=['Budget interval', 'Best 50 (Normal, Diverse)'])
    
    startlow =450
    endlow =1000
    idx=0
    for low in range(startlow, endlow,50):
        
        budget = low+50
        print('-------------------------------------')
        print(budget)
        ones = filter_df(one, low, budget)
        ones.sort_values(by ="points_total", inplace = True, ascending = False)
 #      playerspldata = get.get_players_feature_pl("data/pl_csv/players_raw_", season)
        asdata= load_data.get_data()
        playersdata=get.get_players_feature(asdata)
        all_teams = ones["indexes"].to_list()
        #ss = Counter(flatten(all_teams)).most_common()
       # allpoints= ones['points_total'].to_list() 
        
        #Take 50 best 
        if len(ones)>50:
            best_50 = [ones.iloc[i]['indexes'] for i in range(50)]
        else:
            best_50 = [ones.iloc[i]['indexes'] for i in range(len(ones))]
        
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
            h = get.get_cost_team(playersdata, team)
            
            _ , ret, _ = linR2Inter(h, None, plot)
            #print(r2)
            best_div.append(ret)
            #print(best_div)
        bnor, bdiv = calcpercent(best_div)
        b50  = [bnor,bdiv]
        
        #For printing
        #printpercent('Best 50', best_div, bnor, bdiv, bund)
        
        if useall: 
            diverse=[]
            for team_id in all_teams:
                h = get.get_cost_team(playersdata, team_id)
                _, ret, _ = linR2Inter(h, None, plot)
                diverse.append(ret)
            
            anor, adiv = calcpercent(diverse)
            a  = [anor,adiv]
        
            #For printing
            #printpercent('All', diverse, anor, adiv, aund)
    
            dfres.loc[idx]=[str(low) + ' to ' + str(budget), b50,a]
        else: 
            dfres.loc[idx]=[str(low) + ' to ' + str(budget), b50]
        idx+=1
    
    
    dfres.to_csv('results/as/linperc_' +str(formation)+ '.csv')     


#%%
#ALL PL linperc
#Calc ratio and mean difference
csvfile= 'results/pl/'
seasons=[1617,1718,1819,1920,2021]
perc = 'linperc'
calcratioandmeandifferenceplots(csvfile, seasons, perc)

#%%

#PL noexp linperc
#Calc ratio and mean difference
csvfile= 'results/pl/'
seasons=[1617,1819]
perc = 'noexp/linperc'
calcratioandmeandifferenceplots(csvfile, seasons, perc)

#%%

#PL incnew
#Calc ratio and mean difference
csvfile= 'results/pl/'
seasons=[1617,1819]
perc = 'incnew/linperc'
calcratioandmeandifferenceplots(csvfile, seasons, perc)

#%%

#PL noexp interval
#Calc ratio and mean difference
csvfile= 'results/pl/'
seasons=[1617,1819]
perc = 'noexp/intervalperc'
calcratioandmeandifferenceplots(csvfile, seasons, perc)
#%%
#PL intervalperc
csvfile= 'results/pl/'
seasons=[1617,1718,1819,1920,2021]
perc = 'intervalperc'
calcratioandmeandifferenceplots(csvfile, seasons, perc)
#%%
#AS linperc
csvfile= 'results/as/linperc_' 
seasons=['AS']
perc='linperc'
calcratioandmeandifferenceplots(csvfile, seasons, perc)

#%%
#AS intervalperc
csvfile= 'results/as/intervalperc_' 
seasons=['AS']
perc='intervalperc'
calcratioandmeandifferenceplots(csvfile, seasons, perc)

  #%%
  
def calcratioandmeandifferenceplots(csvfile, seasons, perc):

    formnames= ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]','[4, 4, 2]','[4, 5, 1]','[5, 3, 2]', '[5, 4, 1]']
    
    for season in seasons: 
        b50nortot=np.zeros(11)
        b50divtot=np.zeros(11)
        allanortot=np.zeros(11)
        alladivtot=np.zeros(11)
        
        rationor=np.zeros(11)
        ratiodiv = np.zeros(11)
        
        for forma in formnames: 
            if season != 'AS':
                data = pd.read_csv(csvfile +str(season)+ '/'+ perc+'_' +forma+ '.csv', converters=conv)
            else: 
                data = pd.read_csv(csvfile + forma + '.csv', converters=conv)

            b50 = [ast.literal_eval(d) for d in data['Best 50 (Normal,Diverse)']]
            alla = [ast.literal_eval(d) for d in data['All (Normal,Diverse)']]
    
            b50nortot =[b50nortot[idx]+i for idx,(i,_) in enumerate(b50)]
            b50divtot =[b50divtot[idx]+j for idx,(_,j) in enumerate(b50)]
            allanortot =[allanortot[idx]+i for idx,(i,_) in enumerate(alla)]
            alladivtot =[alladivtot[idx]+j for idx,(_,j) in enumerate(alla)]
            
            b50nor = [i for i,_ in b50]
            b50div = [j for _,j in b50]
            allanor = [i for i,_ in alla]
            alladiv =[j for _,j in alla]
                
            rationor=[rationor[idx]+(a/b) if b!=0 else rationor[idx]+1 for idx,(a,b) in enumerate(zip(b50nor,allanor))]
            ratiodiv=[ratiodiv[idx]+(a/b) if b!=0 else ratiodiv[idx]+1 for idx,(a,b) in enumerate(zip(b50div,alladiv))]
            
        diffnor = [(a-b)/7 for a,b in zip(b50nortot,allanortot)]
        diffdiv = [(a-b)/7 for a,b in zip(b50divtot,alladivtot)]
    
        #b50nortot =[b50nortot[idx]/7 for idx in range(11)]
        #b50divtot =[b50divtot[idx]/7 for idx in range(11)]
        #allanortot =[allanortot[idx]/7 for idx in range(11)]
        #alladivtot =[alladivtot[idx]/7 for idx in range(11)]
        
        X = range(500,1050,50)

        plt.plot(X, diffdiv, '-o')
        plt.plot(X, diffnor, '-o')
        plt.plot(X, np.zeros(11)) 
        plt.xlabel('Budget')
        plt.ylabel('Difference')
        plt.title('Mean difference between percent of all and best 50: ' +str(season))
        plt.legend(labels=['Linear', 'Not linear'])
        plt.xticks(X,range(500,1050, 50))
        plt.savefig('plots/' + perc + '/' + str(season) + 'difference.png', bbox_inches='tight')
    
        plt.show()
        
        rationor = [r/7 for r in rationor]
        ratiodiv = [r/7 for r in ratiodiv]
        plt.plot(X,ratiodiv, '-o')
        plt.plot(X,rationor, '-o')
        plt.legend(labels=['Linear', 'Not linear'])
        plt.title('Mean ratio best 50 and all: '+str(season))
        plt.xticks(X,range(500,1050, 50))
        plt.xlabel('Budget')
        plt.ylabel('Ratio')
        plt.savefig('plots/'+perc+'/' + str(season) + 'ratio.png', bbox_inches='tight')
    
        plt.show()
    
        
       # print(b50nortot)
        #print(allanortot)
        #print(rationor)


#%%
#Intervalperc
#Create df for saving results, checking linearity of all teams, linear or not according to R2

zvalue=3
formations= ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]','[4, 4, 2]','[4, 5, 1]','[5, 3, 2]', '[5, 4, 1]']
useall = True  # T -> alla, F -> bara 50 bästa


for formation in formations: 
    print('Preparing data', str(formation))
    one = pd.read_csv('data_cleaned/as/'+str(formation)+ '.csv', converters =conv)
    print('Done')

    if useall:
        dfres = pd.DataFrame(columns=['Budget interval', 'Best 50 (Normal,Diverse)', 'All (Normal,Diverse)'])
    else: 
        dfres = pd.DataFrame(columns=['Budget interval', 'Best 50 (Normal, Diverse)'])
    
    startlow =450
    endlow =1000
    idx=0
    for low in range(startlow, endlow,50):
        
        budget = low+50
        print('-------------------------------------')
        print(budget)
        ones = filter_df(one, low, budget)
        ones.sort_values(by ="points_total", inplace = True, ascending = False)
        asdata= load_data.get_data()
        playersdata=get.get_players_feature(asdata)
        all_teams = ones["indexes"].to_list()
        
        #Take 50 best 
        if len(ones)>50:
            best_50 = [ones.iloc[i]['indexes'] for i in range(50)]
        else:
            best_50 = [ones.iloc[i]['indexes'] for i in range(len(ones))]
        
        best_div=[]
        i=0
        plot= False
        for team in best_50:
            each_team = helpers_calc_div.team(team,playersdata)
            each_team.create_int()
            each_team.check_int()
            
            if each_team.zero_count <=zvalue:
                
                best_div.append(0)
            else: 
                best_div.append(1)
        bnor, bdiv = calcpercent(best_div)
        b50  = [bnor,bdiv]
                
        if useall: 
            diverse=[]
            print("Calc all teams")
            for team_id in all_teams:
                each_team = helpers_calc_div.team(team_id ,playersdata)
                each_team.create_int()
                each_team.check_int()
                
                if each_team.zero_count <=zvalue:
                    
                    diverse.append(0)
                else: 
                    diverse.append(1) 
            anor, adiv = calcpercent(diverse)
            a  = [anor,adiv]
   
            dfres.loc[idx]=[str(low) + ' to ' + str(budget), b50,a]
        else: 
            dfres.loc[idx]=[str(low) + ' to ' + str(budget), b50]
        idx+=1
    
    
    dfres.to_csv('results/as/intervalperc_' +str(formation)+ '.csv')


#%%
#Intervalperc for noexp
#Create df for saving results, checking linearity of all teams, linear or not according to R2
seasons=[1617,1819]

zvalue=3
formations= ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]','[4, 4, 2]','[4, 5, 1]','[5, 3, 2]', '[5, 4, 1]']
useall = True  # T -> alla, F -> bara 50 bästa

for season in seasons: 
    
    for formation in formations: 
        print('Preparing data', str(formation))
        one = pd.read_csv('data_cleaned/pl/noexp/' +str(season) +'/' +str(formation)+ '.csv', converters =conv)
        print('Done')
    
        if useall:
            dfres = pd.DataFrame(columns=['Budget interval', 'Best 50 (Normal,Diverse)', 'All (Normal,Diverse)'])
        else: 
            dfres = pd.DataFrame(columns=['Budget interval', 'Best 50 (Normal, Diverse)'])
        
        startlow =450
        endlow =1000
        idx=0
        for low in range(startlow, endlow,50):
            
            budget = low+50
            print('-------------------------------------')
            print(budget)
            ones = filter_df(one, low, budget)
            ones.sort_values(by ="points_total", inplace = True, ascending = False)
            playerspldata = get.get_players_feature_pl("data/pl_csv/players_raw_", season)
            all_teams = ones["indexes"].to_list()

            
            #Take 50 best 
            if len(ones)>50:
                best_50 = [ones.iloc[i]['indexes'] for i in range(50)]
            else:
                best_50 = [ones.iloc[i]['indexes'] for i in range(len(ones))]
            
            best_div=[]
            i=0
            plot= False
            for team in best_50:
                each_team = helpers_calc_div.team(team,playerspldata)
                each_team.create_int()
                each_team.check_int()
                
                if each_team.zero_count <=zvalue:
                    
                    best_div.append(0)
                else: 
                    best_div.append(1)
            bnor, bdiv = calcpercent(best_div)
            b50  = [bnor,bdiv]
                    
            if useall: 
                diverse=[]
                print("Calc all teams")
                for team_id in all_teams:
                    each_team = helpers_calc_div.team(team_id ,playerspldata)
                    each_team.create_int()
                    each_team.check_int()
                    
                    if each_team.zero_count <=zvalue:
                        
                        diverse.append(0)
                    else: 
                        diverse.append(1) 
                anor, adiv = calcpercent(diverse)
                a  = [anor,adiv]
       
                dfres.loc[idx]=[str(low) + ' to ' + str(budget), b50,a]
            else: 
                dfres.loc[idx]=[str(low) + ' to ' + str(budget), b50]
            idx+=1
        
        
        dfres.to_csv('results/pl/'+ str(season) +'/noexp/intervalperc_' +str(formation)+ '.csv')

#%%
#Intervalperc for incnew
#Create df for saving results, checking linearity of all teams, linear or not according to R2
seasons=[1617,1819]
seasons=[1819]

zvalue=3
formations= ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]','[4, 4, 2]','[4, 5, 1]','[5, 3, 2]', '[5, 4, 1]']
formations = ['[5, 3, 2]', '[5, 4, 1]']
useall = True  # T -> alla, F -> bara 50 bästa

for season in seasons: 
    
    for formation in formations: 
        print('Preparing data', str(formation))
        one = pd.read_csv('data_cleaned/pl/incnew/' +str(season) +'/' +str(formation)+ '.csv', converters =conv)
        print('Done')
    
        if useall:
            dfres = pd.DataFrame(columns=['Budget interval', 'Best 50 (Normal,Diverse)', 'All (Normal,Diverse)'])
        else: 
            dfres = pd.DataFrame(columns=['Budget interval', 'Best 50 (Normal, Diverse)'])
        
        startlow =450
        endlow =1000
        idx=0
        for low in range(startlow, endlow,50):
            
            budget = low+50
            print('-------------------------------------')
            print(budget)
            ones = filter_df(one, low, budget)
            ones.sort_values(by ="points_total", inplace = True, ascending = False)
            playerspldata = get.get_players_feature_pl("data/pl_csv/players_raw_", season)
            all_teams = ones["indexes"].to_list()

            
            #Take 50 best 
            if len(ones)>50:
                best_50 = [ones.iloc[i]['indexes'] for i in range(50)]
            else:
                best_50 = [ones.iloc[i]['indexes'] for i in range(len(ones))]
            
            best_div=[]
            i=0
            plot= False
            for team in best_50:
                each_team = helpers_calc_div.team(team,playerspldata)
                each_team.create_int()
                each_team.check_int()
                
                if each_team.zero_count <=zvalue:
                    
                    best_div.append(0)
                else: 
                    best_div.append(1)
            bnor, bdiv = calcpercent(best_div)
            b50  = [bnor,bdiv]
                    
            if useall: 
                diverse=[]
                print("Calc all teams")
                for team_id in all_teams:
                    each_team = helpers_calc_div.team(team_id ,playerspldata)
                    each_team.create_int()
                    each_team.check_int()
                    
                    if each_team.zero_count <=zvalue:
                        
                        diverse.append(0)
                    else: 
                        diverse.append(1) 
                anor, adiv = calcpercent(diverse)
                a  = [anor,adiv]
       
                dfres.loc[idx]=[str(low) + ' to ' + str(budget), b50,a]
            else: 
                dfres.loc[idx]=[str(low) + ' to ' + str(budget), b50]
            idx+=1
        
        
        dfres.to_csv('results/pl/'+ str(season) +'/incnew/intervalperc_' +str(formation)+ '.csv')
