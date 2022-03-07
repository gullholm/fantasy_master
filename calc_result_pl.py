# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 14:20:12 2022

@author: jonat

"""

import pandas as pd
import getters
#import calculations as calc
import cleaners
import parsers
import numpy as np
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
import ast

#%%

def clean_all_data_pl(season, bas = "data/pl_csv/players_raw_", dest = "data_cleaned/pl/",  clean_all = True, ns = 3):
    
    playerspldata = getters.get_players_feature_pl(bas, season)
    formations = [[3,4,5],[3,4,5],[1,2,3]]
    form_name = ["df", "mf", "fw"]
    csv_file = str(bas) + str(season) + ".csv"
    all_parts_but_goalie = cleaners.all_forms_as_df_cleaned_pl(csv_file)[1:]
    
    
    for part, df, pos in zip(formations, all_parts_but_goalie, form_name):
        #print(part)
        #print(df)
        print(pos)
        for p in part:
            print(p)
            all_cleaned = cleaners.run_all_cleans(df, p)
            
            if clean_all: 
                print(len(all_cleaned))
                combs = parsers.create_all_combs_from_cleaned_df(playerspldata, all_cleaned, p)
                combs.to_csv(dest + str(season) + "/" + pos + "/" + str(p) + ".csv")
                combs.to_csv(dest + str(season) + "/" + pos + "/" + str(p) + ".csv",index = False)
            else: 
                combs = parsers.create_all_combs_from_cleaned_df(playerspldata, all_cleaned, p)
                combs.to_csv("individual_data_cleaned/pl/" + str(season) + "/" + pos + "/" + str(p) + ".csv",index = False)

    
    # Goalkeepers
    
    gk, df,mf,fw = getters.get_diff_pos(playerspldata)
    
    df_gk = pd.DataFrame.from_dict(gk, orient='index')
    
    sorted_df_gk = df_gk.sort_values(by= ['now_cost'])
    
    cleaned_gk = cleaners.clean_gk(sorted_df_gk)
    cleaned_gk.reset_index(inplace=True)
    cleaned_gk.rename(columns={'index':'indexes'}, inplace=True)
    cleaned_gk.drop('element_type', inplace=True, axis=1)
    if clean_all: 
        cleaned_gk.to_csv("data_cleaned/pl/" + str(season) + "/gk.csv")
    else : 
        cleaned_gk.to_csv("individual_data_cleaned/pl/" + str(season) + "/gk.csv")
        
    print("Done with " + str(season))
    
 # In[]   

# Change for different seasons
seasons = [1617, 1718, 1819, 1920, 2021]
#season = seasons[3]

clean_all = False # if True, clean combinations of players as well

for season in seasons:
    print("cleaning season " + str(season))
    clean_all_data_pl(season)
    
#%%'

clean_all_data_pl('all', ns=3)   
    
# In[]

# saving variables for tessting
season=2021

csv_file = "data/pl_csv/players_raw_" + str(season) + ".csv"
playerspl = pd.read_csv(csv_file) 
playerspl = playerspl.to_dict('index')
playerspldata = getters.get_players_feature_pl(playerspl)

formations = [[3,4,5],[3,4,5],[1,2,3]]
form_name = ["df", "mf", "fw"]
all_parts_but_goalie = cleaners.all_forms_as_df_cleaned_pl(csv_file)[1:]


for part, df, pos in zip(formations, all_parts_but_goalie, form_name):
    #print(part)
    #print(df)
    print(pos)
    for p in part:
        print(p)
        all_cleaned = cleaners.run_all_cleans(df, p)
        combs = parsers.create_all_combs_from_cleaned_df(playerspldata, all_cleaned, p)[0]
        combs.to_csv("data_cleaned/pl/" + str(season) + "/" + pos + "/" + str(p) + ".csv")
        combs.to_csv("data_cleaned/pl/" + str(season) + "/" + pos + "/" + str(p) + ".csv",index = False)


# Goalkeepers

gk, df,mf,fw = getters.get_diff_pos(playerspldata)

df_gk = pd.DataFrame.from_dict(gk, orient='index')

sorted_df_gk = df_gk.sort_values(by= ['now_cost'])

cleaned_gk = cleaners.clean_gk(sorted_df_gk)
cleaned_gk.reset_index(inplace=True)
cleaned_gk.rename(columns={'index':'indexes'}, inplace=True)
cleaned_gk.drop('element_type', inplace=True, axis=1)
cleaned_gk.to_csv("data_cleaned/pl/" + str(season) + "/gk.csv")

print("Done with " + str(season))
   
#%%

def clean_all_data_pl_place_indep(season, bas = "data/pl_csv/players_raw_", dest = "data_cleaned/pl/",  clean_all = True):
    csv_file = str(bas) + str(season) + ".csv"
    playerspl = pd.read_csv(csv_file) 
    playerspl = playerspl.to_dict('index')
    playerspldata = getters.get_players_feature_pl(bas, season)
    all_parts_but_goalie = cleaners.all_forms_as_df_cleaned_pl(bas, season)[1:]
    all_df_but_goalie = pd.concat(all_parts_but_goalie)
    
    gk, df,mf,fw = getters.get_diff_pos(playerspldata)
    df_gk = pd.DataFrame.from_dict(gk, orient='index')
    sorted_df_gk = df_gk.sort_values(by= ['now_cost'])
    
    cleaned_gk = cleaners.clean_gk(sorted_df_gk)
    #cleaned_gk.reset_index(inplace=True)
    #cleaned_gk.rename(columns={'index':'indexes'}, inplace=True)
    cleaned_gk.drop('element_type', inplace=True, axis=1)
    all_df = pd.concat([all_df_but_goalie, cleaned_gk])
    
    all_cleaned = cleaners.run_all_cleans(all_df, 11)
    return(all_cleaned)
    combs = parsers.create_all_combs_from_cleaned_df(playerspldata, all_cleaned[:50], 11)
            #combs.to_csv(dest + str(season) + "/" + pos + "/" + str(p) + ".csv")
            #combs.to_csv(dest + str(season) + "/" + pos + "/" + str(p) + ".csv",index = False)
        
    print("Done with " + str(season))
    return(combs) 

#%%

# BEST ONE, FASTEST ONE

def calculatePositonlessBest(sorttuple, budget, bestPoints, n):

    print('Calculating budget: '+ str(budget))
    sorttuplepoints = sorted([(j,i) for i,j in sorttuple], reverse=True) 
    
    templistA= sorttuplepoints[-n:]
    
    team , teamlist = [], []
    teamPoints, teamCost = 0,0
    allPointsList, allCostsList =[],[]
    
    #Count how many times we are in a loop
    a,b,c,d,e,f,g,h,i,j,k =0,0,0,0,0,0,0,0,0,0,0
    
    start_time = time.time()
    
    while len(templistA) >10:
        a+=1 
        if teamPoints + (sum([i for i,_ in templistA][:11])) < bestPoints:
            break
        if budget < (teamCost + templistA[0][1] + (sum(sorted([j for _,j in templistA])[:10]))):
            templistA.pop(0)
            continue 
        else:
            pl = templistA[0]
            teamPoints+= pl[0]
            teamCost += pl[1]
            team.append(pl)
            templistA.pop(0)
            templistB = templistA.copy()
        
        while len(templistB)>9:
            b+=1
            if teamPoints + (sum([i for i,_ in templistB][:10])) < bestPoints:
                break
            if budget < (teamCost + templistB[0][1] + sum(sorted([j for _,j in templistB])[:9])):
                templistB.pop(0)
                continue 
      
            else:
                tempteamB = team.copy() 
                tempCostB = teamCost.copy()
                tempPointsB = teamPoints.copy()
                pl = templistB[0]
                teamPoints+= pl[0]
                teamCost += pl[1]
                team.append(pl)  
                templistB.pop(0)             
                templistC=templistB.copy()
            
            while len(templistC)>8:
                c+=1
                if teamPoints + (sum([i for i,_ in templistC][:9])) < bestPoints:
                    break
                elif budget < (teamCost + templistC[0][1] + (sum(sorted([j for _,j in templistC])[:8]))):
                    templistC.pop(0)
                    continue 
                else:
                    tempteamC = team.copy()
                    tempCostC = teamCost.copy()
                    tempPointsC = teamPoints.copy()
                    pl = templistC[0]
                    teamPoints+= pl[0]
                    teamCost += pl[1]
                    team.append(pl)
                    templistC.pop(0)
                    templistD=templistC.copy()
            
                while len(templistD)>7:
                    d+=1
                    if teamPoints + (sum([i for i,_ in templistD][:8])) < bestPoints:
                        break
                    if budget < (teamCost + templistD[0][1] + (sum(sorted([j for _,j in templistD])[:7]))):
                        templistD.pop(0)
                        continue 
    
                    else:
                        tempteamD = team.copy() 
                        tempCostD = teamCost.copy()
                        tempPointsD = teamPoints.copy()
                        pl = templistD[0]
                        teamPoints+= pl[0]
                        teamCost += pl[1]
                        team.append(pl)
                        templistD.pop(0)
                        templistE=templistD.copy()
                    
                    while len(templistE)>6:
                        e+=1
                        if teamPoints + (sum([i for i,_ in templistE][:7])) < bestPoints:
                            break
                        if budget < (teamCost + templistE[0][1] + (sum(sorted([j for _,j in templistE])[:6]))):
                            templistE.pop(0)
                            continue 
                       
                        else:
                            tempteamE = team.copy() 
                            tempCostE = teamCost.copy()
                            tempPointsE = teamPoints.copy()
                            pl = templistE[0]
                            teamPoints+= pl[0]
                            teamCost += pl[1]
                            team.append(pl)
                            templistE.pop(0)
                            templistF=templistE.copy()
                        
                        while len(templistF)>5:
                            f+=1
                            if teamPoints + (sum([i for i,_ in templistF][:6])) < bestPoints:
                                break
                            if budget < (teamCost + templistF[0][1] + (sum(sorted([j for _,j in templistF])[:5]))):
                                templistF.pop(0)
                                continue 
                            
    
                            else:
                                tempteamF = team.copy() 
                                tempCostF = teamCost.copy()
                                tempPointsF = teamPoints.copy()
                                pl = templistF[0]
                                teamPoints+= pl[0]
                                teamCost += pl[1]
                                team.append(pl)
                                templistF.pop(0)
                                templistG=templistF.copy()
                        
                            while len(templistG)>4:
                                g+=1
                                if teamPoints + (sum([i for i,_ in templistG][:5])) < bestPoints:
                                    break
                                if budget < (teamCost + templistG[0][1] + (sum(sorted([j for _,j in templistG])[:4]))):
                                    templistG.pop(0)
                                    continue 
                                else:
                                    tempteamG = team.copy() 
                                    tempCostG = teamCost.copy()
                                    tempPointsG = teamPoints.copy()
                                    pl = templistG[0]
                                    teamPoints+= pl[0]
                                    teamCost += pl[1]
                                    team.append(pl)
                                    templistG.pop(0)
                                    templistH=templistG.copy()
                                
                                while len(templistH)>3:
                                    h+=1
                                    if teamPoints + (sum([i for i,_ in templistH][:4])) < bestPoints:
                                        break
                                    if budget < (teamCost + templistH[0][1] + (sum(sorted([j for _,j in templistH])[:3]))):
                                        templistH.pop(0)
                                        continue 
                                    else:
                                        tempteamH = team.copy()
                                        tempCostH = teamCost.copy()
                                        tempPointsH = teamPoints.copy()
                                        pl = templistH[0]
                                        teamPoints+= pl[0]
                                        teamCost += pl[1]
                                        team.append(pl)
                                        templistH.pop(0)
                                        templistI=templistH.copy()
    
                                    
                                    while len(templistI)>2:
                                        i+=1
                                        if teamPoints + (sum([i for i,_ in templistI][:3])) < bestPoints:
                                            break
                                        if budget < (teamCost + templistI[0][1] + (sum(sorted([j for _,j in templistI])[:2]))):
                                            templistI.pop(0)
                                            continue 
                                        else:
                                            tempteamI = team.copy()
                                            tempCostI = teamCost.copy()
                                            tempPointsI = teamPoints.copy()
                                            pl = templistI[0]
                                            teamPoints+= pl[0]
                                            teamCost += pl[1]
                                            team.append(pl)
                                            templistI.pop(0)
                                            templistJ=templistI.copy()
    
                                        
                                        while len(templistJ)>1:
                                            j+=1
                                            if teamPoints + (sum([i for i,_ in templistJ][:2])) < bestPoints:
                                                break
                                            if budget < (teamCost + templistJ[0][1] + (sum(sorted([j for _,j in templistJ])[:1]))):
                                                templistJ.pop(0)
                                                continue 
                                            
                                            else: 
                                                tempteamJ = team.copy() 
                                                tempCostJ = teamCost.copy()
                                                tempPointsJ = teamPoints.copy()
                                                pl = templistJ[0]
                                                teamPoints+= pl[0]
                                                teamCost += pl[1]
                                                team.append(pl)
                                                templistJ.pop(0)
                                                templistK=templistJ.copy()
                                            
    
                                            
                                            while len(templistK)>0:
                                                k+=1
                                                tempteamK = team.copy() 
                                                tempCostK = teamCost.copy()
                                                tempPointsK = teamPoints.copy()
    
                                                if teamPoints + (sum([i for i,_ in templistK][:1])) < bestPoints:
                                                    break                                            
                                                elif teamCost > (budget - templistK[0][1]):
                                                    templistK.pop(0)
                                                    continue
                                                
                                                else:
                                                    pl = templistK[0]
                                                    teamPoints+= pl[0]
                                                    teamCost += pl[1]
                                                    bestPoints = teamPoints 
                                                    
                                                    team.append(pl)
                                    
                                                    allPointsList.append(teamPoints)
                                                    allCostsList.append(teamCost)
                                                    teamlist.append(team)
                                                
                                                templistK.pop(0)
                                                team = tempteamK 
                                                teamCost, teamPoints = tempCostK, tempPointsK                                      
                                            team = tempteamJ 
                                            teamCost, teamPoints = tempCostJ, tempPointsJ
                                            
                                        team = tempteamI    
                                        teamCost, teamPoints = tempCostI, tempPointsI
                                        
                                    team = tempteamH
                                    teamCost, teamPoints = tempCostH, tempPointsH
                                
                                team = tempteamG
                                teamCost, teamPoints = tempCostG, tempPointsG
                        
                            team = tempteamF
                            teamCost, teamPoints = tempCostF, tempPointsF
                                              
                        team = tempteamE
                        teamCost, teamPoints = tempCostE, tempPointsE
                           
                    team = tempteamD
                    teamCost, teamPoints = tempCostD, tempPointsD
                
                team = tempteamC
                teamCost, teamPoints = tempCostC, tempPointsC
            
            team = tempteamB
            teamCost, teamPoints = tempCostB, tempPointsB
        
        teamCost, teamPoints = 0, 0    
        team = []
    
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Total loops: " + str(sum([a,b,c,d,e,f,g,h,i,j,k])))
    
    print('Nr of saved teams: '+ str(len(teamlist)))
    if len(teamlist) ==1: 
        print('Cost: ' + str(max(allCostsList)))
        print('Best points: ' + str(max(allPointsList)))
    else: 
        print('Min cost: ' +str(min(allCostsList)))
        print('Max cost: ' + str(max(allCostsList)))
        print('Lowest saved points: ' + str(min(allPointsList)))
        print('Best points: ' + str(max(allPointsList)))
    return allCostsList, allPointsList, teamlist


#%%

budgets = list(range(500,1050,50))
seasons = [2021]

for season in seasons: 
    print('-------------------------------------------------')
    print('Preprocessing data for season: ' + str(season))

    flat_list = clean_all_data_pl_place_indep(season)   
    tuplist= []
    for tup in flat_list.values:
        tuplist.append((tup[0],tup[1]))
    sorttuple=sorted(tuplist)
    n = len(sorttuple) 
    positionlessdf=pd.DataFrame(columns = ['Budget', 'Best total cost', 'Best total points', 'Sorted individual costs', 'Individual costs'])
    
    bestresults = pd.read_csv('results/pl/' + str(season)+ '/best.csv')
    bestteampoints = bestresults['Best total points'].tolist()
    
    # One possible best that we have achieved that can fasten up the computations:
    if season == 1617: 
        bestteampoints = [1260,1462,1608,1735,1843,1944,2033,2113,2189,2239,2289]    
    if season == 1718:
        bestteampoints= [1277, 1522, 1667,1821, 1939, 2012, 2073, 2127,2166, 2204, 2230]
    if season == 1819:
        bestteampoints= [1319, 1580, 1797, 1950, 2026, 2097, 2150, 2218, 2271, 2307, 2319]
    if season == 1920:
        bestteampoints = [1320, 1573, 1699, 1805, 1904, 1990, 2065, 2127, 2192, 2232, 2273]
    if season == 2021:
        bestteampoints = [1344, 1610, 1768, 1864, 1940, 1993, 2069, 2125, 2165, 2185, 2192]
    dfindex=0
    for idx in range(11):
            
        budget = budgets[idx]
        bestPoints = bestteampoints[idx] 
        bestcost, bestpoints, bestteam = calculatePositonlessBest(sorttuple, budget, bestPoints, n)
        if len(bestteam)==1:
            indcosts = [j for _,j in bestteam[0]]
            sortindcosts  = sorted(indcosts)
            positionlessdf.loc[dfindex] = [budget, bestcost[0], bestpoints[0], sortindcosts,indcosts]
            dfindex+=1
        else:    
            for i, team in enumerate(bestteam):
                indcosts = [j for _,j in team]
                sortindcosts  = sorted(indcosts)
                positionlessdf.loc[dfindex] = [budget, bestcost[i], bestpoints[i], sortindcosts,indcosts]
                dfindex+=1
            
    #positionlessdf.sort_index(inplace=True)
    positionlessdf.to_csv('results/pl/' + str(season) + '/Positionless.csv')

#%%

for season in seasons: 
    budget =450
    bestresults = pd.read_csv('results/pl/'+ str(season) + '/Positionless.csv')
    sortedindcosts = bestresults['Sorted individual costs'].tolist()

    for indcosts in sortedindcosts:
        budget += 50
        h= ast.literal_eval(indcosts)
        
        print('Season is:', season, 'and budget is:', budget)
#%%        
def testNormal(h, plot=True):       
    mu = np.mean(h)
    #mu = np.median(h) # antingen mean eller median... 
    sigma = np.std(h)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    
    high = mu+3*sigma
    low= mu+- 3*sigma
    nrinter = round((high-low)/len(h))
    
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
       # fit = stats.norm.pdf(h, np.mean(h), np.std(h))*2  #this is a fitting indeed
        #plt.plot(h,fit,'-o')
        
        #plt.axvline(x=mu+2*sigma, color='b', ls='--') # *1.645 for 90%
        #plt.axvline(x=mu-2*sigma, color='b', ls='--') # *1.960 for 95 %
        
        #plt.hist(h,nrinter, density = True)
        plt.hist(h,len(h),density=True)
        plt.title(norm1)
        plt.show()
    
    return perc1, ret
     
    
#%%

#Calculate if  normal distributed    

# μ±σ includes approximately 68% of the observations
#μ±2⋅σ includes approximately 95% of the observations
#μ±3⋅σ includes almost all of the observations (99.7% to be more precise)       
        
#%%
import pandas as pd
import numpy as np
import math
import operator
import matplotlib.pyplot as plt

def linR2(h, ax, plot=True):
    
    X=range(len(h))
    Y= h
    
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    
    m = len(X)
    
    # using the formula to calculate m & c
    numer = 0
    denom = 0
    for i in range(m):
      numer += (X[i] - mean_x) * (Y[i] - mean_y)
      denom += (X[i] - mean_x) ** 2
    m = numer / denom
    c = mean_y - (m * mean_x)
    
    # calculating line values x and y
    x = range(len(h))
    y = c + m * x
        
    # calculating R-squared value for measuring goodness of our model. 
    
    ss_t = 0 #total sum of squares
    ss_r = 0 #total sum of square of residuals
    
    for i in range(len(h)): # val_count represents the no.of input x values
      ss_t += (Y[i] - mean_y) ** 2
      ss_r += (Y[i] - y[i]) ** 2
    r2 = 1 - (ss_r/ss_t)
    
    
    if r2> 0.9:
        norm = 'Not normal'
        ret = 0
    else:
        norm = 'Normal'
        ret=1
    #Plot residuals
    #plt.plot(x, [Y[i]-y[i]for i in range(11)], 'o')
    #plt.plot(x,[0]*11,'-')
    #plt.title(norm)
    #plt.show()
    
    if plot:
        ax.plot(x, y, color='r', label='Regression Line')
        ax.scatter(X, Y, c='b', label='Data points')
        ax.set_title(norm)
        ax.legend()
    return r2, ret
    
    
#%%


seasons =[1617, 1718, 1819, 1920, 2021]
i=0   
similar=0 
for season in seasons: 
       
       budget =450
       bestresults = pd.read_csv('results/pl/'+ str(season) + '/Positionless.csv')
       sortedindcosts = bestresults['Sorted individual costs'].tolist()

       for indcosts in sortedindcosts:
           i+=1
           budget += 50
           h= ast.literal_eval(indcosts)
           #h.pop(10)
           #h.pop(0)
           plot=True
           if plot:
               fig, (ax1, ax2) = plt.subplots(1, 2)
               fig.suptitle(i)
               
           val1, ret1 = linR2(h, ax1, plot) #Return 1 if normal
           val2, ret2 = testNormal(h, plot) #Return 1 if normal 
           
           if ret1==ret2:
               print(i, ret1 == ret2)
               similar+=1
           else: 
               print(i, ret1==ret2, ret1, val1, ret2, val2)
               #if val1 > 0.93 or val1 < 0.87: # då äf vi säkra på normal/not normal även om olika
                #   similar+=1
print(similar/i)
#%%
i=0
similar=0
budgets= range(500,1050,50)
h=[]
for budget in budgets:
    bestresults= pd.read_csv('results/pl/budget/' + str(budget)+ '.csv')
    sortedindcosts = bestresults['Sorted individual costs'].tolist()
    h=[]
    for indcosts in sortedindcosts:
        i+=1
        j= ast.literal_eval(indcosts)
        h.extend(j)
    h = sorted(h)    
    plot=True
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(i)
    val1, norm1 = linR2(h, ax1, plot)
    val2, norm2 = testNormal(h, plot)
       
    if norm1==norm2:
       print(i, norm1 == norm2)
       similar+=1
    else: 
       print(i, norm1==norm2, norm1, val1, norm2, val2)
     #  if val1 > 0.93 or val1 < 0.87: # då äf vi säkra på normal/not normal även om olika
      #     similar+=1
print(similar/11)