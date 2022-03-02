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

tuplelist = [(x,y) for x,y in zip(points,costs) ] 
sorttuple = sorted(tuplelist)[::-1]
sortpoints = [i for i,j in sorttuple]
sortcosts = [j for i,j in sorttuple]  

count=0 
bestteampoints =[1170,1397,1565,1705,1824,1930,2020,2107,2170,2231,2249] # for formations
budgets=list(range(500, 1001, 50))
nr = 11

#%%
# try to create all combinations in a tree structure and see how many you get and how fast it will take
# probably in for-loops...


start_time = time.time()

for subset in itertools.combinations(sorttuple, nr):
    count+=1
    teamvalues = [sum(x) for x in zip(*subset)]
    if teamvalues[0]<1800:
        continue
    index = sum(1 for x in budgets if teamvalues[1] > x)
    if 0 < index > 10:
        pass 
    elif teamvalues[0] >= bestteampoints[index]:
        bestteampoints[index] = teamvalues[0]
        print(bestteampoints)
       
    if count%1000000 == 0:
        print(count/1000000)
    if count/1000000 == 10:
        break

print("--- %s seconds ---" % (time.time() - start_time))

       
#%%        
test = list(range(108))
count=0
for sub in itertools.combinations(test, nr):
    count+=1
    # teamvalues = [sum(x) for x in zip(*subset)]
    # if teamvalues[0]<1800:
    #     continue
    # index = sum(1 for x in budgets if teamvalues[1] > x)
    # if 0 < index > 10:
    #    pass 
    # elif teamvalues[0] >= bestteampoints[index]:
    #     bestteampoints[index] = teamvalues[0]
    #     print(bestteampoints)
       
    if count%10000000 == 0:
        print(count/10000000)
        
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
n = 12

templistA = sorttuple[:n]
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
print("Total loops: " + str(sum([a,b,c,d,e,f,g,h,i,j,k])))
