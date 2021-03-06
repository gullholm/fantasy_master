# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 18:12:50 2022

@author: jonat
"""
# In[ ]:

import getters
import numpy as np
import time
import matplotlib.pyplot as plt
import calculations as calc
from calculations import *
import pandas as pd


# Import the data 

data2 = getters.get_data()
players = getters.get_players_feature(data2)
gk, df, mf, fw = getters.get_diff_pos(players)
gk1, def4, mid4, forw2 = calc.createFormation(gk,df,mf,fw, 4, 4, 2, 100)

 
# In[]
# Create cost and points list
costList = createCostList()
pointsList = createPointsList()

# In[ ]:
## Create teams
# Calculate points and cost for all different teams (Time inefficient)    

# n = 50 takes ~130s, n = 30 ~40s

n = 30 # number of players used on each position
teamPointsList, teamCostList, teams =[], [],[]

# Uncomment to test time 
#timeList=[]
#for n in range(30):

start_time = time.time()
for i in gk:
    for j in range(n):
        team0 = np.append(i, forw2[j])
        for k in range(n):
            team1 = np.append(team0, def4[k]) 
            for l in range(n):
                team2 = np.append(team1, mid4[l])
                teams.append(team2)


print("--- %s seconds ---" % (time.time() - start_time))
#runtime = (time.time() - start_time)
#timeList.append(runtime)

start_time = time.time()

for team in teams:
    teamPointsList.append(pointsPerTeam4(team,pointsList))
    teamCostList.append(costPerTeam4(team,costList))

print("--- %s seconds ---" % (time.time() - start_time))
    

calc.printSummary(teamPointsList, teamCostList)
index_max = np.argmax(teamPointsList)
print("Indexes for the best team: " + str(teams[index_max]))




# In[]

#n=100 ~23

# timeList=[] 
# for i in range(10):

n = 100 # number of players used on each position
teamPointsList3 , teamCostList3 = [], []

forPoints, defPoints, midPoints, gkPoints = [], [], [], []
forCost, defCost, midCost, gkCost = [], [], [], []

tpl3app = teamPointsList3.append
tcl3app = teamCostList3.append

tpl3ext = teamPointsList3.extend
tcl3ext = teamCostList3.extend

start_time = time.time()
                
for i in range(n): 
    forPoints.append(pointsPerTeam4(forw2[i],pointsList))
    midPoints.append(pointsPerTeam4(mid4[i],pointsList))        
    defPoints.append(pointsPerTeam4(def4[i],pointsList))
    
    forCost.append(costPerTeam4(forw2[i], costList))
    midCost.append(costPerTeam4(mid4[i], costList))
    defCost.append(costPerTeam4(def4[i], costList))
    
tempCostList, tempPointsList =[], []

papp=tempPointsList.append
capp=tempCostList.append 

for j in range(n):
    totalPoints1 = forPoints[j]
    totalCost1 = forCost[j]
    for k in range(n):
        totalPoints2 = totalPoints1 + defPoints[k]
        totalCost2 = totalCost1 + defCost[k]
        for l in range(n):
            totalPoints3 = totalPoints2 + midPoints[l]
            totalCost3 = totalCost2 + midCost[l]
            papp(totalPoints3)
            capp(totalCost3)           

for i in gk:
    g = pointsList[i-1] 
    h = costList[i-1]
    gkPoints.append(g)
    gkCost.append(h)
    
    totalTeamPoints = [x+g for x in tempPointsList]
    totalTeamCost = [x+h for x in tempCostList]

    tpl3ext(totalTeamPoints)
    tcl3ext(totalTeamCost)

    
print("--- %s seconds ---" % (time.time() - start_time))
    # runtime = (time.time() - start_time)
    # timeList.append(runtime)
    # Results
    # Nr of teams: 1674000
    # Best index: 1512236
    # Best score: 956
    # Total cost for the best team: 813
    # Mean cost: 729
    # Mean points: 572
    
#print(np.mean(timeList)) #2.78

calc.printSummary(teamPointsList3, teamCostList3)

# In[]
# TEST, Little faster but a little off in index? same result...

# timeList=[] 
# for i in range(10):

n = 100 # number of players used on each position ~21s
teamPointsList3 , teamCostList3 = [], []

forPoints, defPoints, midPoints, gkPoints = [], [], [], []
forCost, defCost, midCost, gkPoints = [], [], [], []

tpl3app = teamPointsList3.append
tcl3app = teamCostList3.append

tpl3ext = teamPointsList3.extend
tcl3ext = teamCostList3.extend

start_time = time.time()
                
for i in range(n): 
    forPoints.append(pointsPerTeam4(forw2[i],pointsList))
    midPoints.append(pointsPerTeam4(mid4[i],pointsList))        
    defPoints.append(pointsPerTeam4(def4[i],pointsList))    
    
    forCost.append(costPerTeam4(forw2[i], costList))
    midCost.append(costPerTeam4(mid4[i], costList))
    defCost.append(costPerTeam4(def4[i], costList))
    
tempCostList, tempPointsList =[], []

papp=tempPointsList.append
capp=tempCostList.append 

tpList=[]
tcList=[]
for j in range(n):
    a = defPoints[j]
    b = defCost[j]
    fmPoints = [x+a for x in forPoints]
    fmCost = [x+b for x in forCost]
    tpList.extend(fmPoints)
    tcList.extend(fmCost)

fdmPointsList=[]
fdmCostList=[]

for k in range(n):
    a = midPoints[k]
    b = midCost[k]
    fmdPoints = [x+a for x in tpList]
    fmdCost = [x+b for x in tcList]
    fdmPointsList.extend(fmdPoints)
    fdmCostList.extend(fmdCost)

for i in gk:
    g = pointsList[i-1] 
    h = costList[i-1]
    gkPoints.append(g)

    totalTeamPoints = [x+g for x in fdmPointsList]
    totalTeamCost = [x+h for x in fdmCostList]

    tpl3ext(totalTeamPoints)
    tcl3ext(totalTeamCost)
            

    
print("--- %s seconds ---" % (time.time() - start_time))
    # runtime = (time.time() - start_time)
    # timeList.append(runtime)
    #results
    # Nr of teams: 1674000
    # Best index: 1535610
    # Best score: 956
    # Total cost for the best team: 813
    # Mean cost: 729
    # Mean points: 572
    
#print(np.mean(timeList)) #0.41
calc.printSummary(teamPointsList3, teamCostList3)


# In[]

# Plot some results

plot1 = plt.figure(1)
plt.hist(teamCostList)

plot2 = plt.figure(2)
plt.hist(teamPointsList)

plt.show()

# In[]
# Can be used for testing how much time it will take to run 

xs = list(range(50))
ys = timeList 

print(ys)

p = np.poly1d(np.polyfit(xs ,ys ,deg=2))
print(p)

plt.plot(timeList)
plt.plot(xs, p(xs))

print("Time for 50 is ca: " +str(round(p(50))) + " sec")

print("Time for 88201170 is ca: " +str(round(p(88201170))) + " sec")


# In[]
# want to calculate the maximal value one can get 
# maximal value for 1 gk [154]
# maximal value for 1-2-3-4-5 defenders [166, 327, 476, 622, 764]
# maximal value for 1-2-3-4-5 midfielders [192, 359, 511, 662, 809]
# maximal value for 1-2-3 forwards [154, 307, 446]

#going 4-4-2 starting with goalkeeper-defenders-midfielfers-forwards
#Max value when choosing 11, 10, 9,... 1 more player
# Max value: [1745, 1591, 1445, 1296, 1135, 969, 818, 666, 499, 307, 153]
       

dfPoints = [d.get('total_points') for d in df.values()]
dfCost = [d.get('now_cost') for d in df.values()]
N=5
Nmaxelements(dfPoints, N) 

mfPoints = [d.get('total_points') for d in mf.values()]
mfCost = [d.get('now_cost') for d in mf.values()]
N=5
Nmaxelements(mfPoints, N)

fwPoints = [d.get('total_points') for d in fw.values()]
fwCost = [d.get('now_cost') for d in fw.values()]
N=3
Nmaxelements(fwPoints, N)  

gkPoints = [d.get('total_points') for d in gk.values()]
gkCost = [d.get('now_cost') for d in gk.values()]
    
print((np.max(gkPoints)))
    
# In[]
import getters 
import pandas as pd
import cleaners

def dropRows(df, indexes):
    df = df.drop(indexes, axis=0)
    return df

def del_zeros(sorted_dfs, formation): # Delete #n_part zeros from formation df
    del_sorted_dfs = []
    for (i,df) in enumerate(sorted_dfs):
        dele = list(df.index[(df['total_points'] == 0)][formation[i]:])
        print(dele)
        del_sorted_dfs.append(dropRows(df, dele))
    return(del_sorted_dfs)

sorted_dfs = cleaners.all_forms_as_df_cleaned() # does all cleaning of the whole dataset
bestGK = cleaners.clean_gk(sorted_dfs[0]) # cleans gk so ready for csv


#dfGK = pd.DataFrame.from_dict(gk, orient='index')  
#sorteddfGK = dfGK.sort_values(by=['total_points', 'now_cost'])
#gkDelete = list(sorteddfGK.index[(sorteddfGK['total_points'] == 0)][1:])

#dfDF = pd.DataFrame.from_dict(df, orient='index')  
#sorteddfDF = dfDF.sort_values(by=['total_points', 'now_cost'])
#dfDelete = list(sorteddfDF.index[(sorteddfDF['total_points'] == 0)][4:])

#dfMF = pd.DataFrame.from_dict(mf, orient='index')  
#sorteddfMF = dfMF.sort_values(by=['total_points', 'now_cost'])
#mfDelete = list(sorteddfMF.index[(sorteddfMF['total_points'] == 0)][4:] )

#fFW = pd.DataFrame.from_dict(fw, orient='index')  
#sorteddfFW = dfFW.sort_values(by=['total_points', 'now_cost'])
#fwDelete = list(sorteddfFW.index[(sorteddfFW['total_points'] == 0)][2:])

# In[]



#gkDropZero = dropRows(dfGK, gkDelete)
#dfDropZero = dropRows(dfDF, dfDelete)
#mfDropZero = dropRows(dfMF, mfDelete)
#fwDropZero = dropRows(dfFW, fwDelete)

# In[]    
#Goalkeepers
#First delete all values that generates zero points and are more expensive than
#cheaper ones that generate 0 points
# then just keep the ones that are best for each salary
# then just keep the ones that have less cost but higer points than best one.
# then just keep them who have better points when increasing cost
def saveBetterPointsWhenIncreasingCost(df_part):
    """
    Removes players with higher cost and lower points

    Parameters
    ----------
    df_part : cost and points per formation comb

    Returns
    -------
    df_part : 
        cleaned df
    """
    
    pointsmax=0  
    saveIndexes=[]
    for i in range(df_part.shape[0]):
        if (df_part.iloc[i]['total_points']) > pointsmax:
            pointsmax =  (df_part.iloc[i]['total_points']) 
            saveIndexes.append(df_part.iloc[i].name)
    df_part = df_part.loc[saveIndexes]
    return df_part


    
    
#sorteddfGK = sorted_dfs_del_0[0]
best_GK , bgk_new = clean_gk(sorted_dfs[0])


#idx = sorteddfGK.groupby(['now_cost'])['total_points'].transform(max) == sorteddfGK['total_points']
#gkBestPerSalary = sorteddfGK[idx]

#column = sorteddfGK['total_points']
#max_index = column.idxmax() 
#costForMostPoints = sorteddfGK.loc[max_index]['now_cost']
    
#gkFinal = gkBestPerSalary[gkBestPerSalary['now_cost'] <= costForMostPoints]     

#gkFinalSorted = gkFinal.sort_values(by=['now_cost', 'total_points'], ascending=[True, False])


#Use only these goalkeepers:
#bestGK = saveBetterPointsWhenIncreasingCost(gkFinalSorted)
#bestGK = bestGK.drop(columns = ['element_type'])
#bestGK['indexes'] = gkIndexes
print(bestGK)

# In[]
#Forwards

# G??r s?? vi bara har tv?? per po??ng kvar, tredje p?? en speciell po??ng, 
#Som ??r dyrare kommer man aldrig v??lja

def del_multiple_cost_per_point(sorted_df_part, n):
    """
    delete if there are more than 
    n players that have the same total points
    """
    sorted_df_part = sorted_df_part.sort_values(by=["total_points", "now_cost"])    
    deleteIndexes=[]    

    for i in range(max(sorteddfFW['total_points'])+1):
        if((sorted_df_part['total_points'] == i).sum() > n):
            
            delete = list(sorted_df_part.index[(sorted_df_part['total_points'] == i) ][n:])
            deleteIndexes.extend(delete)
        
    return(dropRows(sorted_df_part,deleteIndexes))

def del_multiple_point_per_cost(sorted_df_part, n):
    """
    delete if there are more than 
    n players that have the same cost
    """

    deleteIndexes=[]    
    sorted_df_part = sorted_df_part.sort_values(by=['now_cost',"total_points"])    
    
    for i in range(max(sorted_df_part['now_cost'])):
        if((sorted_df_part['now_cost'] == i).sum() > n):
            
            fwDelete = list(sorted_df_part.index[(sorted_df_part['now_cost'] == i) ][n:])
            deleteIndexes.extend(fwDelete)
            
    return(dropRows(sorted_df_part,deleteIndexes))

#deleteIndexes=[]
sorteddfFW = sorted_dfs[3]

#for i in range(max(sorteddfFW['total_points'])+1):
#    if((sorteddfFW['total_points'] == i).sum() > 2):
       # print(i)
        
#        fwDelete = list(sorteddfFW.index[(sorteddfFW['total_points'] == i) ][2:])
#        deleteIndexes.extend(fwDelete)
formation = [1,4,4,2]
fwSave2PerPoints = del_multiple_cost_per_point(sorteddfFW,formation[3])        

#deleteIndexes=[]    

sssfw2 = del_multiple_point_per_cost(fwSave2PerPoints, formation[3])


#sortedfw2Points = fwSave2PerPoints.sort_values(by=['now_cost',"total_points"])    

#for i in range(max(fwSave2PerPoints['now_cost'])):
#    if((sortedfw2Points['now_cost'] == i).sum()>2):
#        fwDelete = list(sortedfw2Points.index[(sortedfw2Points['now_cost'] == i) ][:-2])
#        deleteIndexes.extend(fwDelete)
    
#fwFinal = dropRows(sortedfw2Points,deleteIndexes) 
  
# In[]
import numpy as np
from calculations import *
import calculations as calc

def create_all_combs_from_cleaned_df(df_part, form_n):
    
    combs = np.transpose(calc.nump2(len(df_part), form_n))
    combs_indexes = calcIndexOld(combs, df_part.index, form_n, len(combs))  
    pointsList = createPointsList()
    costList = createCostList()
    combsPoints, combsCost = [], []

    for i in range(len(combs)): 
        combsPoints.append(calc.pointsPerTeam4(combs_indexes[i],pointsList))
        combsCost.append(calc.costPerTeam4(combs_indexes[i], costList)) 

    combs_parts = pd.DataFrame(list(zip(combsPoints, combsCost, combs_indexes)),
                           columns =['total_points', 'now_cost', 'indexes'])

    sortedCombs_parts = combs_parts.sort_values(by=['now_cost', 'total_points'], ascending=[True, False])
    
    ssss = cleaners.del_multiple_cost_per_point(sortedCombs_parts,1)        

    sortedCombs_parts = cleaners.del_multiple_point_per_cost(ssss, 1)
    
    return(delete_worse_points_when_increasing_cost(sortedCombs_parts, 1), sortedCombs_parts)

#best_fw = create_all_combs_from_cleaned_df(sssfw2, formation[3])

# In[]

def delete_worse_points_when_increasing_cost(df_part, n_form):
    
    df_part.sort_values(by=['now_cost','total_points'], ascending=[True, False])
    best = df_part.head(n_form)['total_points'].to_list()
    ind_to_del = []
    for i in range(n_form,len(df_part)):
        if df_part.iloc[i]['total_points'] > min(best):
            best.remove(min(best))
            best.append(df_part.iloc[i]['total_points'])
        else: 
            ind_to_del.append(df_part.iloc[i].name)
    return(dropRows(df_part, ind_to_del))
# Midfielders
import cleaners
# G??r s?? vi bara har 4 per po??ng kvar, femte p?? en speciell po??ng, 
#Som ??r dyrare kommer man aldrig v??lja

#for i in range(max(sortedmfDropZero['total_points'])+1):
#    if((sortedmfDropZero['total_points'] == i).sum() > 4):
#        mfDelete = list(sortedmfDropZero.index[(sortedmfDropZero['total_points'] == i) ][4:])
#        deleteIndexes.extend(mfDelete)

#mfSave4PerPoints = dropRows(sortedmfDropZero,deleteIndexes)        
    
#endast 4 b??sta per kostnad
#deleteIndexes=[]    
#sortedmf4Points = mfSave4PerPoints.sort_values(by=['now_cost',"total_points"])    
#for i in range(max(mfSave4PerPoints['now_cost'])):
#     if((sortedmf4Points['now_cost'] == i).sum()>4):
#         mfDelete = list(sortedmf4Points.index[(sortedmf4Points['now_cost'] == i)][:-4])
#         print(mfDelete)
#         deleteIndexes.extend(mfDelete)
#mfFinals2 = del_multiple_cost_per_point(, formation[2])

mfFinals = sorted_dfs[2].sort_values(by=['now_cost','total_points'], ascending=[True, False])

mfFinals1 = del_multiple_point_per_cost(mfFinals, formation[2])

mfFinals2 = del_multiple_cost_per_point(mfFinals1, formation[2])

mffff_final = delete_worse_points_when_increasing_cost(mfFinals2, formation[2])

mf_combs = create_all_combs_from_cleaned_df(mffff_final, formation[2])


# In[]
import cleaners
sorted_dfs = cleaners.all_forms_as_df_cleaned()
mfFinalSort = sorted_dfs[2].sort_values(by=['now_cost','total_points'], ascending=[True, False])
manDelIndexes = []


    
mffff_final = delete_worse_points_when_increasing_cost(mfFinalSort, formation[2])
            
# Index p?? de som inte ??r tillr??ckligt bra att v??lja om man f??r en miljon till
# det ??r b??ttre att beh??lla en billigare som ger mer po??ng
# om det kommer ett v??rde som ??r l??gre ??n det fj??rde h??gsta v??rdet kan man ta bort den

# fourthBest=0
# Po??ng f??r de fyra billigaste:
#Best=[0, 0,0,0]
#for i in range(4,len(mfFinalSort)):
#    if mfFinalSort.iloc[i]['total_points'] > min(Best):
#        Best.remove(min(Best))
#        Best.append(mfFinalSort.iloc[i]['total_points'])
#    else: 
#        manDelIndexes.append(mfFinalSort.iloc[i].name)
#manmfFinal = dropRows(mfFinalSort, manDelIndexes)



mf_combs = create_all_combs_from_cleaned_df(mffff_final, formation[2])

#%%
"""
DEFENDERS:
"""
dfFinals = cleaners.all_forms_as_df_cleaned()[1]

dfFinals = del_multiple_point_per_cost(mfFinals, formation[1])

dfFinals = del_multiple_cost_per_point(mfFinals, formation[1])

dffff_final = delete_worse_points_when_increasing_cost(mfFinalSort, formation[2])

df_combs = create_all_combs_from_cleaned_df(mffff_final, formation[2])




# In[] OOOOOOOOOLD
# calculate all possible combinations 

"""
midf = np.transpose(calc.nump2(len(manmfFinal), 4))
n=len(midf)
midfielders = calcIndexOld(midf, manmfFinal.index, 4, n)

midPoints, midCost = [], []

for i in range(n): 
    midCost.append(costPerTeam4(midfielders[i], costList)) 
    
mfPanda = pd.DataFrame(list(zip(midPoints, midCost, midfielders)),
               columns =['total_points', 'now_cost', 'indexes'])

sortedCostmfPanda= mfPanda.sort_values(by=['now_cost', 'total_points'], ascending=[True, False])

bestMF, mfIndexes= saveBetterPointsWhenIncreasingCost(sortedCostmfPanda)
print(bestMF)
print(len(bestMF))
"""

# In[]

# Defenders

# G??r s?? vi bara har 4 per po??ng kvar, femte p?? en speciell po??ng, 
#Som ??r dyrare kommer man aldrig v??lja

"""
sorteddfDropZero= dfDropZero.sort_values(by=[ 'total_points', 'now_cost'])    
deleteIndexes=[]
for i in range(max(sorteddfDropZero['total_points'])+1):
    if((sorteddfDropZero['total_points'] == i).sum() > 4):
        #print(i)
        
        dfDelete = list(sorteddfDropZero.index[(sorteddfDropZero['total_points'] == i) ][4:])
        deleteIndexes.extend(dfDelete)

#Tar manuellt bort den med minuspo??ng ocks?? 
deleteIndexes.append(276)
dfSave4PerPoints = dropRows(sorteddfDropZero,deleteIndexes)        

#endast 4 b??sta per kostnad
deleteIndexes=[]    
sorteddf4Points = dfSave4PerPoints.sort_values(by=['now_cost',"total_points"])    
for i in range(max(dfSave4PerPoints['now_cost'])):
     if((sorteddf4Points['now_cost'] == i).sum()>4):
         dfDelete = list(sorteddf4Points.index[(sorteddf4Points['now_cost'] == i)][:-4])
         print(dfDelete)
         deleteIndexes.extend(dfDelete)
    
dfFinal = dropRows(sorteddf4Points,deleteIndexes) 
"""
# In[]
print(dfFinal.sort_values(by=['total_points', 'now_cost']))

dfFinalSort = dfFinal.sort_values(by=['now_cost','total_points'], ascending=[True, False])
manDelIndexes = []

# Index p?? de som inte ??r tillr??ckligt bra att v??lja om man f??r en miljon till
# det ??r b??ttre att beh??lla en billigare som ger mer po??ng
# om det kommer ett v??rde som ??r l??gre ??n det fj??rde h??gsta v??rdet kan man ta bort den

fourthBest=0
Best=[0, 34,12,12]
for i in range(4,len(dfFinalSort)):

    if dfFinalSort.iloc[i]['total_points'] > min(Best):
        Best.remove(min(Best))
        Best.append(dfFinalSort.iloc[i]['total_points'])
        #print(Best)
    else: 
        #print(dfFinalSort.iloc[i].name)
        manDelIndexes.append(dfFinalSort.iloc[i].name)
    
mandfFinal = dropRows(dfFinal, manDelIndexes)

# In[]
# Calculate all possible combinations 
defe = np.transpose(calc.nump2(len(mandfFinal), 4))
n=len(defe)
defenders = calcIndexOld(defe, mandfFinal.index, 4, n)

defPoints, defCost= [], []

for i in range(n): 
    defPoints.append(pointsPerTeam4(defenders[i],pointsList))
    defCost.append(costPerTeam4(defenders[i], costList)) 
    
dfPanda = pd.DataFrame(list(zip(defPoints, defCost, defenders)),
               columns =['total_points', 'now_cost', 'indexes'])

sortedCostdfPanda= dfPanda.sort_values(by=['now_cost', 'total_points'], ascending=[True, False])

bestDF, dfIndexes = saveBetterPointsWhenIncreasingCost(sortedCostdfPanda)

print(bestDF)
print(len(bestDF))

# inte sparat om de har samma kostnad , samma b??sta... kanske man borde
# isf >= ist??llet f??r > n??r man j??mf??r  


# In[]
#??ndra s?? det ??r lika i alla
bestGK=bestGK[['total_points', 'now_cost', 'Goalkeeper']]

# In[]

bestGK.to_csv('1_goalkeeper.csv', index=False)
bestDF.to_csv('4_defenders.csv', index=False)
bestMF.to_csv('4_midfielders.csv', index=False)
bestFW.to_csv('2_forwards.csv', index=False)