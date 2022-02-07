# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 11:42:44 2022

@author: jonat
"""

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

dfGK = pd.DataFrame.from_dict(gk, orient='index')  
sorteddfGK = dfGK.sort_values(by=['total_points', 'now_cost'])
#gkDelete = list(sorteddfGK.index[(sorteddfGK['total_points'] == 0)][1:])

dfDF = pd.DataFrame.from_dict(df, orient='index')  
sorteddfDF = dfDF.sort_values(by=['total_points', 'now_cost'])
dfDelete = list(sorteddfDF.index[(sorteddfDF['total_points'] == 0)][4:])

dfMF = pd.DataFrame.from_dict(mf, orient='index')  
sorteddfMF = dfMF.sort_values(by=['total_points', 'now_cost'])
mfDelete = list(sorteddfMF.index[(sorteddfMF['total_points'] == 0)][4:] )

dfFW = pd.DataFrame.from_dict(fw, orient='index')  
sorteddfFW = dfFW.sort_values(by=['total_points', 'now_cost'])
#fwDelete = list(sorteddfFW.index[(sorteddfFW['total_points'] == 0)][2:])

# In[]

def dropRows(df, indexes):
    df = df.drop(indexes, axis=0)
    return df

#gkDropZero = dropRows(dfGK, gkDelete)
dfDropZero = dropRows(dfDF, dfDelete)
mfDropZero = dropRows(dfMF, mfDelete)
#fwDropZero = dropRows(dfFW, fwDelete)

# In[]    
#Goalkeepers
#First delete all values that generates zero points and are more expensive than
#cheaper ones that generate 0 points
# then just keep the ones that are best for each salary
# then just keep the ones that have less cost but higer points than best one.
# then just keep them who have better points when increasing cost
    
idx = sorteddfGK.groupby(['now_cost'])['total_points'].transform(max) == sorteddfGK['total_points']
gkBestPerSalary= sorteddfGK[idx]

column = sorteddfGK['total_points']
max_index = column.idxmax() 
costForMostPoints = sorteddfGK.loc[max_index]['now_cost']
    
gkFinal = gkBestPerSalary[gkBestPerSalary['now_cost'] <= costForMostPoints]     

gkFinalSorted = gkFinal.sort_values(by=['now_cost', 'total_points'], ascending=[True, False])

def saveBetterPointsWhenIncreasingCost(df):
    pointsmax=0  
    saveIndexes=[]   
    for i in range(df.shape[0]):
        if (df.iloc[i]['total_points']) > pointsmax:
            pointsmax =  (df.iloc[i]['total_points'])       
            saveIndexes.append(df.iloc[i].name)
  
    return df.loc[saveIndexes], saveIndexes

#Use only these goalkeepers:
bestGK, gkIndexes = saveBetterPointsWhenIncreasingCost(gkFinalSorted)
bestGK = bestGK.drop(columns = ['element_type'])
bestGK['Goalkeeper'] = gkIndexes
print(bestGK)

# In[]
#Forwards

# Gör så vi bara har två per poäng kvar, tredje på en speciell poäng, 
#Som är dyrare kommer man aldrig välja

deleteIndexes=[]
for i in range(max(sorteddfFW['total_points'])+1):
    if((sorteddfFW['total_points'] == i).sum() > 2):
       # print(i)
        
        fwDelete = list(sorteddfFW.index[(sorteddfFW['total_points'] == i) ][2:])
        deleteIndexes.extend(fwDelete)

fwSave2PerPoints = dropRows(sorteddfFW,deleteIndexes)        
 
deleteIndexes=[]    
sortedfw2Points = fwSave2PerPoints.sort_values(by=['now_cost',"total_points"])    
for i in range(max(fwSave2PerPoints['now_cost'])):
    if((sortedfw2Points['now_cost'] == i).sum()>2):
        fwDelete = list(sortedfw2Points.index[(sortedfw2Points['now_cost'] == i) ][:-2])
        deleteIndexes.extend(fwDelete)
    
fwFinal = dropRows(sortedfw2Points,deleteIndexes) 
  
# In[]

forw = np.transpose(calc.nump2(len(fwFinal), 2))
n=len(forw)
forwards = calcIndexOld(forw, fwFinal.index, 2, n)  

forPoints, forCost = [], []

for i in range(n): 
    forPoints.append(pointsPerTeam4(forwards[i],pointsList))
    forCost.append(costPerTeam4(forwards[i], costList)) 
    
fwPanda = pd.DataFrame(list(zip(forPoints, forCost, forwards)),
               columns =['total_points', 'now_cost', '2forw'])

sortedCostfwPanda= fwPanda.sort_values(by=['now_cost', 'total_points'], ascending=[True, False])

bestFW, fwIndexes= saveBetterPointsWhenIncreasingCost(sortedCostfwPanda)

print(bestFW)
print(len(bestFW))

# In[]

# Midfielders

# Gör så vi bara har 4 per poäng kvar, femte på en speciell poäng, 
#Som är dyrare kommer man aldrig välja
sortedmfDropZero= mfDropZero.sort_values(by=[ 'total_points', 'now_cost'])    
deleteIndexes=[]
for i in range(max(sortedmfDropZero['total_points'])+1):
    if((sortedmfDropZero['total_points'] == i).sum() > 4):
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


# In[]
print(mfFinal.sort_values(by=['total_points', 'now_cost']))

mfFinalSort = mfFinal.sort_values(by=['now_cost','total_points'], ascending=[True, False])
manDelIndexes = []

# Index på de som inte är tillräckligt bra att välja om man får en miljon till
# det är bättre att behålla en billigare som ger mer poäng
# om det kommer ett värde som är lägre än det fjärde högsta värdet kan man ta bort den

fourthBest=0
# Poäng för de fyra billigaste:
Best=[0, 0,15,33]
for i in range(4,len(mfFinalSort)):

    if mfFinalSort.iloc[i]['total_points'] > min(Best):
        Best.remove(min(Best))
        Best.append(mfFinalSort.iloc[i]['total_points'])
        #print(Best)
    else: 
        #print(mfFinalSort.iloc[i].name)
        manDelIndexes.append(mfFinalSort.iloc[i].name)
    #print(mfFinal.iloc[i]['total_points'])
    
#manDelIndexes.extend([269, 279, 451, 337, 207, 293, 96, 124, ])
manmfFinal = dropRows(mfFinal, manDelIndexes)


# In[]
# calculate all possible combinations 
midf = np.transpose(calc.nump2(len(manmfFinal), 4))
n=len(midf)
midfielders = calcIndexOld(midf, manmfFinal.index, 4, n)

midPoints, midCost = [], []

for i in range(n): 
    midPoints.append(pointsPerTeam4(midfielders[i],pointsList))
    midCost.append(costPerTeam4(midfielders[i], costList)) 
    
mfPanda = pd.DataFrame(list(zip(midPoints, midCost, midfielders)),
               columns =['total_points', 'now_cost', '4mid'])

sortedCostmfPanda= mfPanda.sort_values(by=['now_cost', 'total_points'], ascending=[True, False])

bestMF, mfIndexes= saveBetterPointsWhenIncreasingCost(sortedCostmfPanda)
print(bestMF)
print(len(bestMF))


# In[]

# Defenders

# Gör så vi bara har 4 per poäng kvar, femte på en speciell poäng, 
#Som är dyrare kommer man aldrig välja
sorteddfDropZero= dfDropZero.sort_values(by=[ 'total_points', 'now_cost'])    
deleteIndexes=[]
for i in range(max(sorteddfDropZero['total_points'])+1):
    if((sorteddfDropZero['total_points'] == i).sum() > 4):
        #print(i)
        
        dfDelete = list(sorteddfDropZero.index[(sorteddfDropZero['total_points'] == i) ][4:])
        deleteIndexes.extend(dfDelete)

#Tar manuellt bort den med minuspoäng också 
deleteIndexes.append(276)
dfSave4PerPoints = dropRows(sorteddfDropZero,deleteIndexes)        

#endast 4 bästa per kostnad
deleteIndexes=[]    
sorteddf4Points = dfSave4PerPoints.sort_values(by=['now_cost',"total_points"])    
for i in range(max(dfSave4PerPoints['now_cost'])):
     if((sorteddf4Points['now_cost'] == i).sum()>4):
         dfDelete = list(sorteddf4Points.index[(sorteddf4Points['now_cost'] == i)][:-4])
         print(dfDelete)
         deleteIndexes.extend(dfDelete)
    
dfFinal = dropRows(sorteddf4Points,deleteIndexes) 

# In[]
print(dfFinal.sort_values(by=['total_points', 'now_cost']))

dfFinalSort = dfFinal.sort_values(by=['now_cost','total_points'], ascending=[True, False])
manDelIndexes = []

# Index på de som inte är tillräckligt bra att välja om man får en miljon till
# det är bättre att behålla en billigare som ger mer poäng
# om det kommer ett värde som är lägre än det fjärde högsta värdet kan man ta bort den

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
               columns =['total_points', 'now_cost', '4def'])

sortedCostdfPanda= dfPanda.sort_values(by=['now_cost', 'total_points'], ascending=[True, False])

bestDF, dfIndexes = saveBetterPointsWhenIncreasingCost(sortedCostdfPanda)

print(bestDF)
print(len(bestDF))

# inte sparat om de har samma kostnad , samma bästa... kanske man borde
# isf >= istället för > när man jämför  


# In[]
#ändra så det är lika i alla
bestGK=bestGK[['total_points', 'now_cost', 'Goalkeeper']]

# In[]

bestGK.to_csv('1_goalkeeper.csv', index=False)
bestDF.to_csv('4_defenders.csv', index=False)
bestMF.to_csv('4_midfielders.csv', index=False)
bestFW.to_csv('2_forwards.csv', index=False)