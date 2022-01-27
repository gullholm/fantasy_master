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

# Import the data 

data2 = getters.get_data()
players = getters.get_players_feature(data2)
gk, df, mf, fw = getters.get_diff_pos(players)
def4, mid4, forw2 = calc.createFormation(4, 4, 2, 100)

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
    gk

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
                
for i in range(n): 
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

