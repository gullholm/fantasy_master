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
import itertools
import time

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

flat_list = clean_all_data_pl_place_indep(1617)   
 

#%%
points =[]
costs = []
for i in range(len(flat_list)): 
    points.append(flat_list.iloc[i]['total_points'])
    costs.append(flat_list.iloc[i]['now_cost'])

tuplelist = [(x,y) for x,y in zip(costs,points) ] 
sorttuple = sorted(tuplelist)
sortcosts = [i for i,j in sorttuple]
sortpoints = [j for i,j in sorttuple]  
        
#%%

#Works for range and faster , half as many loops
    
k = 11
n = 23
start_time = time.time()

a = np.ones((k, n-k+1), dtype=np.int8)
a[0] = np.arange(n-k+1)
for j in range(1, k):
    reps = (n-k+j) - a[j-1]
    a = np.repeat(a, reps, axis=1)
    ind = np.add.accumulate(reps)
    a[j, ind[:-1]] = 1-reps[1:]
    a[j, 0] = j
    a[j] = np.add.accumulate(a[j])

correctcase = a.T   
print("--- %s seconds ---" % (time.time() - start_time))

templistA = list(range(n))
countB = 0
team , teamlist = [], []


#Count how many times we are in a loop
a,b,c,d,e,f,g,h,i,j,k =0,0,0,0,0,0,0,0,0,0,0

start_time = time.time()

while len(templistA) >10:
    a+=1
    team.append(templistA[0])
    templistA.pop(0)
    templistB = templistA.copy()
    
    while len(templistB)>9:
        b+=1
        tempteamB = team.copy() 
        team.append(templistB[0])  
        templistB.pop(0)             
        templistC=templistB.copy()
        
        while len(templistC)>8:
            c+=1
            tempteamC = team.copy() 
            team.append(templistC[0])
            templistC.pop(0)
            templistD=templistC.copy()
        
            while len(templistD)>7:
                d+=1
                tempteamD = team.copy() 
                team.append(templistD[0])
                templistD.pop(0)
                templistE=templistD.copy()
                
                while len(templistE)>6:
                    e+=1
                    tempteamE = team.copy() 
                    team.append(templistE[0])
                    templistE.pop(0)
                    templistF=templistE.copy()
                    
                    while len(templistF)>5:
                        f+=1
                        tempteamF = team.copy() 
                        team.append(templistF[0])
                        templistF.pop(0)
                        templistG=templistF.copy()
                    
                        while len(templistG)>4:
                            g+=1
                            tempteamG = team.copy() 
                            team.append(templistG[0])
                            templistG.pop(0)
                            templistH=templistG.copy()
                            
                            while len(templistH)>3:
                                h+=1
                                tempteamH = team.copy() 
                                team.append(templistH[0])
                                templistH.pop(0)
                                templistI=templistH.copy()

                                
                                while len(templistI)>2:
                                    i+=1
                                    tempteamI = team.copy() 
                                    team.append(templistI[0])
                                    templistI.pop(0)
                                    templistJ=templistI.copy()

                                    
                                    while len(templistJ)>1:
                                        j+=1
                                        tempteamJ = team.copy() 
                                        team.append(templistJ[0])
                                        templistJ.pop(0)
                                        templistK=templistJ.copy()

                                        
                                        while len(templistK)>0:
                                            k+=1
                                            tempteamK = team.copy() 
                                            team.append(templistK[0])

                                            teamlist.append(team)
                                            templistK.pop(0)
                                            team = tempteamK 
                                        
                                        team = tempteamJ 
                                    
                                    team = tempteamI    
                                
                                team = tempteamH
                            
                            team = tempteamG
                    
                        team = tempteamF
                                          
                    team = tempteamE
                       
                team = tempteamD
            
            team = tempteamC
            
        team = tempteamB

    team = []

print("--- %s seconds ---" % (time.time() - start_time))
print('All values are correct: ' + str((teamlist == correctcase).all()))
print("Total loops: " + str(sum([a,b,c,d,e,f,g,h,i,j,k])))
        

#%%

# For our values
n = 107
budgets = list(range(500,1050,50))
idx= 9
budget = budgets[idx]
lowerbudget = budget - 50 

#Best points for any fixed formation, we assume best free is better than 
# with a formation
bestteampoints = [1170,1397,1565,1705,1824,1930,2020,2107,2170,2231,2249]

# One possible best that we have achieved that can fasten up the computations:
bestteampoints = [1170,1397,1565,1705,1824,1930,2033,2113,2189,2239,2289]    
bestPoints = bestteampoints[idx] 
#costlist = sorted([i for i,_ in sorttuple[:n]])
#constraints = [budget-sum(costlist[:i+1]) for i in range(11)]

 


templistA = sorttuple[-n:]
templistA= sorttuple[:n]
pointsUsed = sorted([j for _,j in templistA], reverse=True)
bestPointsLeft = [sum(pointsUsed[:i]) for i in range(1,12)]
team , teamlist = [], []
teamPoints, teamCost = 0,0
allPointsList, allCostsList =[],[]

#Count how many times we are in a loop
a,b,c,d,e,f,g,h,i,j,k =0,0,0,0,0,0,0,0,0,0,0



start_time = time.time()

while len(templistA) >10:
    a+=1
    if teamCost > (budget - (sum(sorted([i for i,_ in templistA])[:11]))):
        break
    #elif teamCost + (sum(sorted([i for i,_ in templistA])[-11:])) < lowerbudget:
   #     break      
    elif teamPoints < (bestPoints- bestPointsLeft[10]):
        break
    else:
        pl = templistA[0]
        teamPoints+= pl[1]
        teamCost += pl[0]
        team.append(pl)
        templistA.pop(0)
        templistB = templistA.copy()
    
    while len(templistB)>9:
        b+=1
        if teamCost > (budget - (sum(sorted([i for i,_ in templistB])[:10]))):
            break
        elif teamPoints < (bestPoints- bestPointsLeft[9]):
            break
        #elif teamCost + (sum(sorted([i for i,_ in templistA])[-10:])) < lowerbudget:
         #   break
        else:
            tempteamB = team.copy() 
            tempCostB = teamCost.copy()
            tempPointsB = teamPoints.copy()
            pl = templistB[0]
            teamPoints+= pl[1]
            teamCost += pl[0]
            team.append(pl)  
            templistB.pop(0)             
            templistC=templistB.copy()
        
        while len(templistC)>8:
            c+=1
            if teamCost > (budget - (sum(sorted([i for i,_ in templistC])[:9]))):
                break
            elif teamPoints < (bestPoints- bestPointsLeft[8]):
                break
           # elif teamCost + (sum(sorted([i for i,_ in templistA])[-9:])) < lowerbudget:
            #    break
            else:
                tempteamC = team.copy()
                tempCostC = teamCost.copy()
                tempPointsC = teamPoints.copy()
                pl = templistC[0]
                teamPoints+= pl[1]
                teamCost += pl[0]
                team.append(pl)
                templistC.pop(0)
                templistD=templistC.copy()
        
            while len(templistD)>7:
                d+=1
                if teamCost > (budget - (sum(sorted([i for i,_ in templistD])[:8]))):
                    break
                elif teamPoints < (bestPoints- bestPointsLeft[7]):
                    break
             #   elif teamCost + (sum(sorted([i for i,_ in templistA])[-8:])) < lowerbudget:
              #      break
                else:
                    tempteamD = team.copy() 
                    tempCostD = teamCost.copy()
                    tempPointsD = teamPoints.copy()
                    pl = templistD[0]
                    teamPoints+= pl[1]
                    teamCost += pl[0]
                    team.append(pl)
                    templistD.pop(0)
                    templistE=templistD.copy()
                
                while len(templistE)>6:
                    e+=1
                    if teamCost > (budget - (sum(sorted([i for i,_ in templistE])[:7]))):
                        break
                    elif teamPoints < (bestPoints- bestPointsLeft[6]):
                        break
                 #   elif teamCost + (sum(sorted([i for i,_ in templistA])[-7:])) < lowerbudget:
                  #      break
                    else:
                        tempteamE = team.copy() 
                        tempCostE = teamCost.copy()
                        tempPointsE = teamPoints.copy()
                        pl = templistE[0]
                        teamPoints+= pl[1]
                        teamCost += pl[0]
                        team.append(pl)
                        templistE.pop(0)
                        templistF=templistE.copy()
                    
                    while len(templistF)>5:
                        f+=1
                        if teamCost > (budget - (sum(sorted([i for i,_ in templistF])[:6]))):
                            break
                        elif teamPoints < (bestPoints- bestPointsLeft[5]):
                            break
                   #     elif teamCost + (sum(sorted([i for i,_ in templistA])[-6:])) < lowerbudget:
                    #        break
                        else:
                            tempteamF = team.copy() 
                            tempCostF = teamCost.copy()
                            tempPointsF = teamPoints.copy()
                            pl = templistF[0]
                            teamPoints+= pl[1]
                            teamCost += pl[0]
                            team.append(pl)
                            templistF.pop(0)
                            templistG=templistF.copy()
                    
                        while len(templistG)>4:
                            g+=1
                            if teamCost > (budget - (sum(sorted([i for i,_ in templistG])[:5]))):
                                break
                            elif teamPoints < (bestPoints- bestPointsLeft[4]):
                                break
                     #       elif teamCost + (sum(sorted([i for i,_ in templistA])[-5:])) < lowerbudget:
                      #          break
                            else:
                                tempteamG = team.copy() 
                                tempCostG = teamCost.copy()
                                tempPointsG = teamPoints.copy()
                                pl = templistG[0]
                                teamPoints+= pl[1]
                                teamCost += pl[0]
                                team.append(pl)
                                templistG.pop(0)
                                templistH=templistG.copy()
                            
                            while len(templistH)>3:
                                h+=1
                                if teamCost > (budget - (sum(sorted([i for i,_ in templistH])[:4]))):
                                    break
                                elif teamPoints < (bestPoints- bestPointsLeft[3]):
                                    break
                       #         elif teamCost + (sum(sorted([i for i,_ in templistA])[-4:])) < lowerbudget:
                        #            break
                                else:
                                    tempteamH = team.copy()
                                    tempCostH = teamCost.copy()
                                    tempPointsH = teamPoints.copy()
                                    pl = templistH[0]
                                    teamPoints+= pl[1]
                                    teamCost += pl[0]
                                    team.append(pl)
                                    templistH.pop(0)
                                    templistI=templistH.copy()

                                
                                while len(templistI)>2:
                                    i+=1
                                    if teamCost > (budget - (sum(sorted([i for i,_ in templistI])[:3]))):
                                        break
                                    elif teamPoints < (bestPoints- bestPointsLeft[2]):
                                        break
                         #           elif teamCost + (sum(sorted([i for i,_ in templistA])[-3:])) < lowerbudget:
                          #              break
                                    else:
                                        tempteamI = team.copy()
                                        tempCostI = teamCost.copy()
                                        tempPointsI = teamPoints.copy()
                                        pl = templistI[0]
                                        teamPoints+= pl[1]
                                        teamCost += pl[0]
                                        team.append(pl)
                                        templistI.pop(0)
                                        templistJ=templistI.copy()

                                    
                                    while len(templistJ)>1:
                                        j+=1
                                        if teamCost > (budget - (sum(sorted([i for i,_ in templistJ])[:2]))):
                                            break
                                        elif teamPoints < (bestPoints- bestPointsLeft[1]):
                                            break
                           #             elif teamCost + (sum(sorted([i for i,_ in templistA])[-2:])) < lowerbudget:
                            #                break
                                        else: 
                                            tempteamJ = team.copy() 
                                            tempCostJ = teamCost.copy()
                                            tempPointsJ = teamPoints.copy()
                                            pl = templistJ[0]
                                            teamPoints+= pl[1]
                                            teamCost += pl[0]
                                            team.append(pl)
                                            templistJ.pop(0)
                                            templistK=templistJ.copy()
                                        

                                        
                                        while len(templistK)>0:
                                            k+=1
                                            tempteamK = team.copy() 
                                            tempCostK = teamCost.copy()
                                            tempPointsK = teamPoints.copy()
                                            
                                            if teamCost > (budget - (sum(sorted([i for i,_ in templistK])[:1]))):
                                                break
                                            elif teamPoints < (bestPoints- bestPointsLeft[0]):
                                                break
                             #               elif teamCost + (sum(sorted([i for i,_ in templistA])[-1:])) < lowerbudget:
                              #                  break
                                            elif teamPoints + templistK[0][1] < bestPoints:
                                                pass

                                            else:
                                                pl = templistK[0]
                                                teamPoints+= pl[1]
                                                teamCost += pl[0]
                                                
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

# l_el = [el for el in allCostsList if el > budget]
# nr_el = len(l_el)
# print(nr_el)

print(min(allCostsList))
print(max(allCostsList))
print(min(allPointsList))
print(max(allPointsList))


# no budgetcap, n=23 , tot loops: 2496143, time: 14 , #allcosts: 1352078

#%%

# Go for sorted points 

#Kan inte kolla de billigaste utan måste kolla de 11 sista , kan gå


n = 107
budgets = list(range(500,1050,50))
idx= 0
budget = budgets[idx]
#lowerbudget = budget - 50 

#Best points for any fixed formation, we assume best free is better than 
# with a formation
bestteampoints = [1170,1397,1565,1705,1824,1930,2020,2107,2170,2231,2249]

# One possible best that we have achieved that can fasten up the computations:
#bestteampoints = [1170,1397,1565,1705,1824,1930,2033,2113,2189,2239,2289]    

bestPoints = bestteampoints[idx] 
#costlist = sorted([i for i,_ in sorttuple[:n]])
#constraints = [budget-sum(costlist[:i+1]) for i in range(11)]

sorttuplepoints = sorted([(j,i) for i,j in sorttuple], reverse=True) 


templistA = sorttuplepoints[-n:]
templistA= sorttuplepoints[:n]
pointsUsed = sorted([i for i,_ in templistA], reverse=True)
bestPointsLeft = [sum(pointsUsed[:i]) for i in range(1,12)]

costsUsed = ([j for _,j in templistA][-11:])
cheapestCostsLeft = [sum(costsUsed[:i]) for i in range(1,12)]



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
    if teamCost > (budget - cheapestCostsLeft[10]):
        break 
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
        if teamCost > (budget - cheapestCostsLeft[9]):
            break
  
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
            elif teamCost > (budget - cheapestCostsLeft[8]):
                break
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
                elif teamCost > (budget - cheapestCostsLeft[7]):
                    break

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
                    elif teamCost > (budget - cheapestCostsLeft[6]):
                        break
                   
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
                        elif teamCost > (budget - cheapestCostsLeft[5]):
                            break
                        

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
                            elif teamCost > (budget - cheapestCostsLeft[4]):
                                break
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
                                elif teamCost > (budget - cheapestCostsLeft[3]):
                                    break
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
                                    elif teamCost > (budget - cheapestCostsLeft[2]):
                                        break
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
                                        elif teamCost > (budget - cheapestCostsLeft[1]):
                                            break
                                        
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
                                            elif teamCost > (budget - cheapestCostsLeft[0]):
                                                break

                                            elif teamPoints + templistK[0][1] < bestPoints:
                                                pass

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

# l_el = [el for el in allCostsList if el > budget]
# nr_el = len(l_el)
# print(nr_el)

print(min(allCostsList))
print(max(allCostsList))
print(min(allPointsList))
print(max(allPointsList))


# no budgetcap, n=23 , tot loops: 2496143, time: 14 , #allcosts: 1352078
