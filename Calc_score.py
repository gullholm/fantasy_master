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
from calculations import pointsPerTeam4, createCostList, createPointsList, costPerTeam4


# Import the data 

data2 = getters.get_data()
players = getters.get_players_feature(data2)
gk, df, mf, fw = getters.get_diff_pos(players)
gk1, def4, mid4, forw2 = calc.createFormation(4, 4, 2, 3005)

 

# In[ ]:
## Create teams
# Calculate points and cost for all different teams (Time inefficient)    
from calculations import pointsPerTeam4, createCostList, createPointsList, costPerTeam4

# n = 50 takes ~130s, n = 30 ~40s

n = 30 # number of players used on each position
teamPointsList=[]
teamCostList=[]
teams=[]

costList = calc.createCostList()
pointsList = calc.createPointsList()


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
# importing the functions from calculations
#from calculations import pointsPerTeam4, createCostList, createPointsList, costPerTeam4

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

costList = createCostList()
pointsList = createPointsList()

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

# importing the functions from calculations
#from calculations import pointsPerTeam4, createCostList, createPointsList, costPerTeam4

#n=100 ~21s

# timeList=[] 
# for i in range(10):

n = 100 # number of players used on each position
teamPointsList3 , teamCostList3 = [], []

forPoints, defPoints, midPoints, gkPoints = [], [], [], []
forCost, defCost, midCost, gkPoints = [], [], [], []

tpl3app = teamPointsList3.append
tcl3app = teamCostList3.append

tpl3ext = teamPointsList3.extend
tcl3ext = teamCostList3.extend

costList = createCostList()
pointsList = createPointsList()

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
# 50~2s

print("Time for 88201170 is ca: " +str(round(p(88201170))) + " sec")
# for the moment 15.6 million years...
# update, 1.6 million years...
# update 200 000 - 300 000 years

# In[]

#things tried to spped up the loop 
# sum, np.sum , tuple, list, importing the function from calculations
# not nestling loops
import time
start_time = time.time()
                
forPoints, midPoints, defPoints, forCost, midCost, defCost = [],[], [], [], [],[]

for i in range(200): 
    forPoints.append(pointsPerTeam4(forw2[i],pointsList))
    midPoints.append(pointsPerTeam4(mid4[i],pointsList))        
    defPoints.append(pointsPerTeam4(def4[i],pointsList))
    
    forCost.append(costPerTeam4(forw2[i], costList))
    midCost.append(costPerTeam4(mid4[i], costList))
    defCost.append(costPerTeam4(def4[i], costList))

forPoints_np = np.asarray(forPoints)
midPoints_np = np.asarray(midPoints)
defPoints_np = np.asarray(defPoints)
gkPoints_np = np.asarray(gkPoints)


fm = np.add.outer(forPoints_np, midPoints_np)
fmd = np.add.outer(fm, defPoints_np)
fmdg = np.add.outer(fmd, gkPoints_np)

print("--- %s seconds ---" % (time.time() - start_time))

# In[]
#Best keeper in points and points per cost is index: 374
# for i in gk:
#     g = pointsList[i-1] 
#     h = costList[i-1]
#     print(str(i) + "has score: " + str(g) +" and cost: " + str(h))
#     print(g/h)
    
#     totalTeamPoints = [x+g for x in fdmPointsList]
#     totalTeamCost = [x+h for x in fdmCostList]

#     tpl3ext(totalTeamPoints)
#     tcl3ext(totalTeamCost)
    
#totalTeamPoints = [x+pointsList[374-1] for x in fdmPointsList]
#totalTeamCost = [x+costList[374-1] for x in fdmCostList]  
#tpl3ext(totalTeamPoints)
#tcl3ext(totalTeamCost)


# In[]
# want to calculate the maximal value one can get 
# maximal value for 1 gk [154]
# maximal value for 1-2-3-4-5 defenders [166, 327, 476, 622, 764]
# maximal value for 1-2-3-4-5 midfielders [192, 359, 511, 662, 809]
# maximal value for 1-2-3 forwards [154, 307, 446]

#going 4-4-2 starting with goalkeeper-defenders-midfielfers-forwards
#Max value when choosing 11, 10, 9,... 1 more player
# Max value: [1745, 1591, 1445, 1296, 1135, 969, 818, 666, 499, 307, 153]

def Nmaxelements(list1, N):
    final_list = []
  
    for i in range(0, N): 
        max1 = 0
          
        for j in range(len(list1)):     
            if list1[j] > max1:
                max1 = list1[j];
                  
        list1.remove(max1);
        final_list.append(max1)
          
    print(final_list)
    
   

#dfPoints=[]
#dfCost=[]
#for player in df.values() :
#    dfPoints.append(player['total_points'])
#    dfCost.append(player['now_cost'])
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
print(len(gkPoints))

# In[]
dfPoints = {k:v.get('total_points') for (k,v) in df.items()}
dfPointz = np.array(list(dfPoints.items()))

dfPoints = [d.get('total_points') for d in df.values()]

# In[]
# Pseudocode
# data cleaning 
# gk ,def,def,def,def,...


# if satser
# for i in alla mlvalkte
#     for ... alla försvarare:
#         for alla försvarare som är kvar(-1):
#             for alla försv som ör kvar(-2): 
#                 if värde < best-nuvarande värde:
#                     break
#                 else: fortsätt
                
# for i gk
#     for combination av def4 :
#         if värde < best-nuvarande värde:
#             break
#         else: fortsätt
#             for combination av 4 midf:
#                 if värde < best-nuvarande värde:
#                     break
#                 else: fortsätt
#                 for combination 2 forw
               

# ? - budget 750
# 300+307=607

# [23,234...] -
# starta från den och byt ut? ...
    
# In[]
import pandas as pd

deleteIndexList = []
dfGK = pd.DataFrame.from_dict(gk, orient='index')  
sorteddfGK = dfGK.sort_values(by=['total_points', 'now_cost'])

gkDelete = list(sorteddfGK.index[(sorteddfGK['total_points'] == 0)][1:])
print(gkDelete)
deleteIndexList.extend(gkDelete)


dfDF = pd.DataFrame.from_dict(df, orient='index')  
sorteddfDF = dfDF.sort_values(by=['total_points', 'now_cost'])

dfDelete=list(sorteddfDF.index[(sorteddfDF['total_points'] == 0)][4:])
print(dfDelete)
deleteIndexList.extend(sorteddfDF.index[(sorteddfDF['total_points'] == 0)][4:])


dfMF = pd.DataFrame.from_dict(mf, orient='index')  
sorteddfMF = dfMF.sort_values(by=['total_points', 'now_cost'])

mfDelete = list(sorteddfMF.index[(sorteddfMF['total_points'] == 0)][4:] )
print(mfDelete)
deleteIndexList.extend(list(sorteddfMF.index[(sorteddfMF['total_points'] == 0)][4:]))


dfFW = pd.DataFrame.from_dict(fw, orient='index')  
sorteddfFW = dfFW.sort_values(by=['total_points', 'now_cost'])

fwDelete = list(sorteddfFW.index[(sorteddfFW['total_points'] == 0) ][2:])
print(fwDelete)
deleteIndexList.extend(list(sorteddfFW.index[(sorteddfFW['total_points'] == 0) ][2:]))

print(deleteIndexList)
deleteIndexList442 = deleteIndexList
print(len(deleteIndexList))

# Start 4-4-2, 5.001283427967741222*10^20 
# now 6.4678340676557608 * 10^19

# In[]

def dropRows(df, indexes):
    df = df.drop(indexes, axis=0)
    return df

gkDropZero = dropRows(dfGK, gkDelete)
dfDropZero = dropRows(dfDF, dfDelete)
mfDropZero = dropRows(dfMF, mfDelete)
fwDropZero = dropRows(dfFW, fwDelete)

# In[]    

#First delete all values that generates zero points and are more expensive than
#cheaper ones that generate 0 points
# then just keep the ones that are best for each salary
# then just keep the ones that have less score but higer cost than best one.
# then just keep them who have better points when increasing cost

#print(gkDropZero.sort_values(by=[ 'now_cost','total_points']))
    
idx = gkDropZero.groupby(['now_cost'])['total_points'].transform(max) == gkDropZero['total_points']
gkBestPerSalary = gkDropZero[idx]

#print(gkBestPerSalary.sort_values(by= ['now_cost']))

column = gkDropZero["total_points"]
max_index = column.idxmax() 
costForMostPoints = gkDropZero.loc[max_index]['now_cost']
    
gkFinal = gkBestPerSalary[gkBestPerSalary['now_cost'] <= costForMostPoints]     
    
#print(gkFinal.sort_values(by=['now_cost']))

pointsmax=0  
saveIndexes=[]   
for i in range(gkFinal.shape[0]):
    if (gkFinal.sort_values(by=['now_cost']).iloc[i]['total_points']) > pointsmax:
        pointsmax =  (gkFinal.sort_values(by=['now_cost']).iloc[i]['total_points'])       
        saveIndexes.append(gkFinal.sort_values(by=['now_cost']).iloc[i].name)

#Use only these goalkeepers: 
bestGK = gkFinal.loc[saveIndexes]
print(bestGK)

# In[]

# Gör så vi bara har två per poäng kvar, tredje på en speciell poäng, 
#Som är dyrare kommer man aldrig välja
sortedfwDropZero= fwDropZero.sort_values(by=[ 'total_points', 'now_cost'])    
deleteIndexes=[]
for i in range(max(sortedfwDropZero['total_points'])+1):
    if((sortedfwDropZero['total_points'] == i).sum() > 2):
       # print(i)
        
        fwDelete = list(sortedfwDropZero.index[(sortedfwDropZero['total_points'] == i) ][2:])
        deleteIndexes.extend(fwDelete)

fwSave2PerPoints = dropRows(sortedfwDropZero,deleteIndexes)        
    
 
deleteIndexes=[]    
sortedfw2Points = fwSave2PerPoints.sort_values(by=['now_cost',"total_points"])    
for i in range(max(fwSave2PerPoints['now_cost'])):
    if((sortedfw2Points['now_cost'] == i).sum()>2):
        fwDelete = list(sortedfw2Points.index[(sortedfw2Points['now_cost'] == i) ][:-2])
        deleteIndexes.extend(fwDelete)
    
fwFinal = dropRows(sortedfw2Points,deleteIndexes)  
    
    
  
    # In[]

def calcIndexOld(indexlist, dat, nr, length):
    returnlist=[]
    for i in range(length):
        temp = []
        for j in range(nr):    
            temp.append(list(dat)[indexlist[i][j]])
        returnlist.append(temp)
    return returnlist


forw = np.transpose(calc.nump2(len(fwFinal), 2))

n=len(forw)

forwards = calcIndexOld(forw, fwFinal.index, 2, n)  

forPoints = []
forCost = []

costList = createCostList()
pointsList = createPointsList()

for i in range(n): 
    forPoints.append(pointsPerTeam4(forwards[i],pointsList))
    forCost.append(costPerTeam4(forwards[i], costList)) 
    
fwPanda = pd.DataFrame(list(zip(forPoints, forCost, forwards)),
               columns =['total_points', 'now_cost', '2forw'])

sortedCostfwPanda= fwPanda.sort_values(by=['now_cost'])

pointsmax=0  
saveIndexes=[]   
for i in range(sortedCostfwPanda.shape[0]):
    if (sortedCostfwPanda.iloc[i]['total_points']) > pointsmax:
        pointsmax =  (sortedCostfwPanda.iloc[i]['total_points'])       
        saveIndexes.append(sortedCostfwPanda.iloc[i].name)
        #print(sortedCostfwPanda.iloc[i].name)
#Use only these forward combinations: 
bestFW = sortedCostfwPanda.loc[saveIndexes]
print(bestFW)
print(len(bestFW))


# bästa tvåmannalaget man kan skapa har summa 158 i kostnad


# In[]

# Midfielders


# Gör så vi bara har 4 per poäng kvar, femte på en speciell poäng, 
#Som är dyrare kommer man aldrig välja
sortedmfDropZero= mfDropZero.sort_values(by=[ 'total_points', 'now_cost'])    
deleteIndexes=[]
for i in range(max(sortedmfDropZero['total_points'])+1):
    if((sortedmfDropZero['total_points'] == i).sum() > 4):
        #print(i)
        
        mfDelete = list(sortedmfDropZero.index[(sortedmfDropZero['total_points'] == i) ][4:])
        deleteIndexes.extend(mfDelete)

mfSave4PerPoints = dropRows(sortedmfDropZero,deleteIndexes)        
    
#endast 4 bästa per kostnad
deleteIndexes=[]    
sortedmf4Points = mfSave4PerPoints.sort_values(by=['now_cost',"total_points"])    
for i in range(max(mfSave4PerPoints['now_cost'])):
     if((sortedmf4Points['now_cost'] == i).sum()>4):
         mfDelete = list(sortedmf4Points.index[(sortedmf4Points['now_cost'] == i)][:-4])
         print(mfDelete)
         deleteIndexes.extend(mfDelete)
    
mfFinal = dropRows(sortedmf4Points,deleteIndexes) 



midf = np.transpose(calc.nump2(len(mfFinal), 4))

n=len(midf)
# In[]
start_time = time.time()
midfielders = calcIndexOld(midf, mfFinal.index, 4, 200000)
print("--- %s seconds ---" % (time.time() - start_time))

# In[]
print(mfFinal.sort_values(by=['total_points', 'now_cost']))

manDelIndexes = []

# index på de som inte ens är top 4 om man har obegränsat med pengar, 
# dvs kommer aldrig bli valda
manDelIndexes.extend([310,241,15,106])

# 


manmfFinal = dropRows(mfFinal, manDelIndexes)

