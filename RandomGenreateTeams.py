# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 13:22:56 2022

@author: jonade
"""

# generate random integer values
from random import *
import pandas as pd
import getters
import calculations
import cleaners
import ast
import matplotlib.pyplot as plt
generic = lambda x: ast.literal_eval(x)
import numpy as np
import random 
from random import choice
import collections

#%%

#Random generate teams from ditribution or randomly.

def generateRandomTeam(allpositions, budget, formation):
        
    gkcost, gkpoints, gkindCost = addPositions(allpositions[0], 1)
    dfcost, dfpoints, dfindCost = addPositions(allpositions[0], formation[0])
    mfcost, mfpoints, mfindCost = addPositions(allpositions[0], formation[1])
    fwcost, fwpoints, fwindCost = addPositions(allpositions[0], formation[2])
    
    teamCost = gkcost + dfcost + mfcost + fwcost 
    
    lowerBudget= 500
    if teamCost <= budget and teamCost >= lowerBudget:
        teamPoints = gkpoints + dfpoints + mfpoints + fwpoints
        teamDynamics =  gkindCost + dfindCost + mfindCost + fwindCost
        return teamCost, teamPoints, teamDynamics
    else:
        return None, None, None
    

def addPositions(df, n):
    indexes = sample(list(df.keys()), n)
    cost = 0
    points = 0
    indCost = [] 
    for idx in indexes:
        cost += df[idx]['now_cost']
        points += df[idx]['total_points'] 
        indCost.append(df[idx]['now_cost'])
    
    return cost, points, indCost    
        
def generateRandomTeamFromCleaned(budget,formation, combs_all):

    tPoints, tCost = 0, 0 
    tIndexes = []

    for comb in combs_all:
        row = comb.loc[choice(comb.index)]
        tPoints += + row['total_points']
        tCost += row['now_cost']
        tIndexes.extend(row['indexes'])        
    lower = 700
    if tCost <= budget and tCost >= lower:    
        return tCost, tPoints, tIndexes     
    else: 
        return None, None, None
    
def saveAllCombs(season, formation):
    
    conv = {'indexes': generic}
    df = formation[0]
    mf = formation[1]
    fw = formation[2]
    
    df_csv = "data_cleaned/pl/" + str(season) + "/df/" + str(df) + ".csv"
    mf_csv = "data_cleaned/pl/" + str(season) + "/mf/" + str(mf) + ".csv"
    fw_csv = "data_cleaned/pl/" + str(season) + "/fw/" + str(fw) + ".csv"
    
    gk_combs = pd.read_csv("data_cleaned/pl/" + str(season) + "/gk.csv", converters = conv)
    df_combs = pd.read_csv(df_csv, converters = conv)
    mf_combs = pd.read_csv(mf_csv, converters = conv)
    fw_combs = pd.read_csv(fw_csv, converters = conv)
    gk_combs['indexes'] = gk_combs['indexes'].apply(lambda x: [x])
    gk_combs.drop('Unnamed: 0', axis=1, inplace=True)
    all_combs = [gk_combs, df_combs, mf_combs, fw_combs]
    
    return all_combs
        
# In[]
season = 1617
csv_file = "data/pl_csv/players_raw_" + str(season) + ".csv"
playerspl = pd.read_csv(csv_file) 
playerspl = playerspl.to_dict('index')
playerspldata = getters.get_players_feature_pl('data/pl_csv/players_raw_',season)

allpositions = getters.get_diff_pos(playerspldata)

allCosts, allPoints, allDynamics =[], [], []

budget = 550
formations = [[3,5,2],[3,4,3],[4,4,2],[4,3,3],[4,5,1],[5,3,2],[5,4,1]]

while len(allCosts) < 10000:
    formation = choice(formations)
    teamCost, teamPoints, teamDynamics = generateRandomTeam(allpositions, budget, formation)
    
    if teamCost is not None:
        allCosts.append(teamCost)
        allPoints.append(teamPoints)
        allDynamics.append(teamDynamics)

print(max(allCosts))

#%%
plotHistOfAllCostsAndPoints(allCosts, allPoints, budget, "random")

# In[]
budget = 1000
formations = [[3,5,2],[3,4,3],[4,4,2],[4,3,3],[4,5,1],[5,3,2],[5,4,1]]

allCosts, allPoints, allDynamics =[], [], []
season = 1617

allCombsPerFormation = []
for formation in formations:
    allCombsPerFormation.append(saveAllCombs(season, formation))

while len(allCosts) < 1000:
    idx = randint(0, len(formations)-1)
    formation = formations[idx]
    allCombs = allCombsPerFormation[idx] 
    teamCost, teamPoints, teamDynamics = generateRandomTeamFromCleaned(budget, formation, allCombs)
    
    if teamCost is not None:
        allCosts.append(teamCost)
        allPoints.append(teamPoints)
        allDynamics.append(teamDynamics)

# In[]
# for understanding
def printAndPlotSummary(allCosts, allPoints, allDynamics, budget): 
    max_value = max(allPoints)
    max_index = allPoints.index(max_value)
    #print('Best index: ' +str(max_index))
    print('Best total points: ' +str(max_value))
    print('Total cost for best team: ' + str(allCosts[max_index]))
    #print(allDynamics[max_index])
    print("Individual costs: " +str(getters.get_cost_team_pl(playerspl, allDynamics[max_index])))
    print("Players in best team: " + str(getters.get_full_name_team_pl(playerspl, allDynamics[max_index])))
    print("Mean total points: " + str(sum(allPoints)/len(allPoints)))
    print("Mean total cost: " + str(sum(allCosts)/len(allCosts)))


# In[]
plotHistOfAllCostsAndPoints(allCosts, allPoints, budget, "Random")

# In[]

def cleanAllPositions(season):
    csv_file = "data/pl_csv/players_raw_" + str(season) + ".csv"
    playerspl = pd.read_csv(csv_file) 
    playerspl = playerspl.to_dict('index')
    playerspldata = getters.get_players_feature_pl("data/pl_csv/players_raw_", season)
    
    formations = [[3,4,5],[3,4,5],[1,2,3]]
    form_name = ["df", "mf", "fw"]
    all_parts_but_goalie = cleaners.all_forms_as_df_cleaned_pl("data/pl_csv/players_raw_",season)[1:]
    
    individualCleansPerPosition =[]
    
    for part, df, pos in zip(formations, all_parts_but_goalie, form_name):
        print(pos)
        for p in part:
            print(p)
            all_cleaned = cleaners.run_all_cleans(df, p)  
            individualCleansPerPosition.append(all_cleaned)
            
    
    # Goalkeepers
    
    gk, _,_,_ = getters.get_diff_pos(playerspldata)
    
    df_gk = pd.DataFrame.from_dict(gk, orient='index')
    
    sorted_df_gk = df_gk.sort_values(by= ['now_cost'])
    
    cleaned_gk = cleaners.clean_gk(sorted_df_gk)
    cleaned_gk.reset_index(inplace=True)
    cleaned_gk.rename(columns={'index':'indexes'}, inplace=True)
    cleaned_gk.drop('element_type', inplace=True, axis=1)
    
    individualCleansPerPosition.append(cleaned_gk)
    
    print("Done with " + str(season))
    return individualCleansPerPosition



#%%

# minimum salary, in our case 38 # maybe cheapest in best team
# total budget, in our case 700 # can change
# C = 700/11 ~63.64, a = 37
#spanning the range from a to 2C-a

#theory_m is first not None value
def calculateTheoryDistribution(theList, budget):
    n = 11
    theory_m = next(idx for idx,item in zip(range(len(theList)),theList) if item is not None)
    #TEST
    #theory_m += 30
    
    C = budget/n
    theory_expensive = round(2*C-theory_m)  
    theory_k = (theory_expensive-theory_m)/(n-1)
    for x in range(11):
        low = round(theory_k*x+theory_m-theory_k/2)
        high = round(theory_k*x+theory_m+theory_k/2)
        print(low)
        print(high)
    return theory_k, theory_m

def generateTeamFromDistribution(theList, budget, theory_k, theory_m):
    
    n = 11
    totalPoints, totalCost = 0, 0
    teamDistr = []
    for x in range(n):
        low = round(theory_k*x+theory_m-theory_k/2)
        high = round(theory_k*x+theory_m+theory_k/2)
        
        if x == 0: 
            low = theory_m
        cost = randint(low, high)
        while cost >= len(theList):
             cost=randint(low,high)
        #print('cost', cost)     
        templist = theList[cost]
        while (templist is None): 
            cost=randint(low,high)
            while cost >= len(theList):
                 cost=randint(low,high) 
            #print('cost:', cost)     
            templist = theList[cost]
        
        teamDistr.append(cost)    
        totalPoints += choice(templist) # add points
        totalCost += cost
    
    return totalPoints, totalCost, teamDistr

#%%
budget = 900
allCosts, allPoints, allDynamics =[], [], []

theory_k, theory_m = calculateTheoryDistribution(theList, budget)

while (len(allCosts)<10000):
    points, costs, dynamics = generateTeamFromDistribution(theList, budget, theory_k, theory_m)
    
    if costs < budget:
        allCosts.append(costs)
        allPoints.append(points)
        allDynamics.append(dynamics)

# Plot results distribution
plotHistOfAllCostsAndPoints(allCosts, allPoints, budget, "distribution")
print('Sum of 50 best', sum(sorted(allPoints)[-50:]))
print('Mean of 50 best', sum(sorted(allPoints)[-50:])/50)     

#%%
print(allDynamics[0])
plt.plot(allDynamics[0], 'o') 

#%%

def generateRandomTeamFromAllPlayers(theList, budget): 
    n=11
    
    res = [i for i in range(len(theList)) if theList[i] is not None]
    totalPoints, totalCost = 0, 0
    teamDistr = []
    
    for i in range(n):
        cost = choice(res)    
        templist = theList[cost]
        while not templist: 
            cost = choice(res)    
            templist = theList[cost]
        teamDistr.append(cost)
        totalPoints += choice(templist) # add points
        totalCost += cost
    
    return totalPoints, totalCost, teamDistr

#%%
budget = 800
lowerbudget = 770
allCosts, allPoints, allDynamics =[], [], []

while (len(allCosts)<10000):
    points, costs, dynamics = generateRandomTeamFromAllPlayers(theList, budget)
    if costs < budget and costs > lowerbudget:
        allCosts.append(costs)
        allPoints.append(points)
        allDynamics.append(dynamics)

plotHistOfAllCostsAndPoints(allCosts, allPoints, budget, "random")        
#%%
# Plot results

def plotHistOfAllCostsAndPoints(allCosts, allPoints, budget, title, season):        
    plt.hist(allCosts)
    plt.title("Season: " + str(season) + ", Budget: " + str(budget) + ", Distr: " + title)
    plt.xlabel("Costs")
    plt.ylabel("Amount")
    plt.show()
    
    plt.hist(allPoints)    
    plt.title("Season: " + str(season) + ", Budget: " + str(budget) + ", Distr: " + title) 
    plt.xlabel("Points")
    plt.ylabel("Amount")       
    plt.show()
    
#%%   
 
cleaned = cleanAllPositions(1819)
#%%
#Cleaned players for 5 def, 5 mid, 3 forw, 1 keeper

values=[2,5,8,9]
cleanedPlayers = []
for i in range(len(cleaned)):
    if i in values:
        cleanedPlayers.append(cleaned[i])

#%%
#gk,df,mf,fw = getters.get_diff_pos(playerspldata)

#for g in gk.items():
 #   print(g[1]['now_cost'])
    
#def splitDfByCost():
 #   return-    
#%%

printAndPlotSummary(allCosts, allPoints, allDynamics, budget)

#%%

# Create a combinations of all seasons in PL 

def combineAllSeasonsPl():
    
    seasons=[1617, 1718,1819,1920,2021]
    for season in seasons: 
        csv_file = "data/pl_csv/players_raw_" + str(season) + ".csv"
        playerspl = pd.read_csv(csv_file) 
        playerspl = playerspl.to_dict('index')
        playerspldata = getters.get_players_feature_pl(playerspl)

        sortIdxByCost = sorted(playerspldata, key=lambda k: (playerspldata[k]['now_cost']))
        
        test_dictionary = { i : playerspldata[idx] for idx, i in zip(sortIdxByCost, range(len(sortIdxByCost))) }
        highest_cost = test_dictionary[len(test_dictionary)-1]['now_cost']
        print(test_dictionary[len(test_dictionary)-1]['total_points'])
        value = -1
        if season == 1617:
            theList = [None]*133 #132 highest value for all costs in pl
            templist=[]
            for key in test_dictionary.values(): 
                if key['now_cost'] <= value:
                    templist.append(key['total_points'])
                #print(key)
                else: 
                    theList[value]= templist  
                    templist = []
                    value = key['now_cost']
                    templist.append(key['total_points'])
                    if key['now_cost'] == 127:
                        theList[127] = templist
            print(theList[132])           
        else:
            value= -1
            for key in test_dictionary.values(): 
                if key['now_cost'] <= value:
                    templist.append(key['total_points'])
                #print(key)
                else: 
                    if theList[value] is None:
                        theList[value]= templist 
                    else:    
                        beforeList = theList[value]
                        newList = beforeList + templist
                        theList[value]= newList
                    templist = []
                    value = key['now_cost']
                    templist.append(key['total_points'])
                    if season == 1718:
                        if key['now_cost'] == 131:
                            if theList[131] is None:
                                theList[131]= templist 
                            else: 
                                beforeList = theList[131]
                                newList = beforeList + templist
                                theList[131]= newList
                            print(theList[132])        
                    elif season == 1819:        
                        if key['now_cost'] == 132:
                            if theList[132] is None:
                                theList[132]= templist 
                            else: 
                                  
                                beforeList = theList[132]
                                newList = beforeList + templist
                                theList[132]= newList
                            print(theList[132])       
                    elif season == 1920:        
                        if key['now_cost'] == 125:
                           if theList[125] is None:
                               theList[125]= templist 
                           else: 
                               beforeList = theList[125]
                               newList = beforeList + templist
                               theList[125]= newList
                           print(theList[132])   
                    else:        
                        if key['now_cost'] == 129:
                          if theList[129] is None:
                              theList[129]= templist 
                          else: 
                              beforeList = theList[129]
                              newList = beforeList + templist
                              theList[129]= newList 
                          print(theList[132])
    theList[132]=[259]    # fixade manuellt för inget funka                   
    return theList
        
    
#%%
#Klart
PLCombined = combineAllSeasonsPl()  

#%%

for lists in PLCombined: 
    if lists is not None: 
        sortlists = lists.sort() 
        print(lists[-11:])
        
#%%
seasons=[1617,1718,1819,1920,2021]
all_filenames = ["data/pl_csv/players_raw_" + str(season) + ".csv" for season in seasons]
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
add= 0
newIDs = []
for idxes, rows in combined_csv.iterrows():
    
    if idxes == 0:
        add += 1000 
    #print(rows['id']+add)
    newIDs.append(rows['id']+add)
    
    #print(rows['id'])
combined_csv['id'] = newIDs
combined_csv.to_csv( "players_raw_all.csv", index=False, encoding='utf-8-sig')        


#%%

# Costs 38 to 127 in season 1617 - interval of 89, little right scew is ok
def generateRandomNormal(budget, mu, sigma,step, possibleCosts):
    #mu = round(budget/11)
    #sigma = round((mu-38)/1.645)
    #step=round(2*sigma/5)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    
    # random generate
    test=[] 
    #7 intervals with distr 1 1 2 3 2 1 1
    intervals= [1, 1, 2, 3, 2, 1, 1]
    binstest=[]
    
    for i, idx in enumerate(intervals):
        if i == 0 :
            templist=[]
            for j in range(38, mu-sigma-1):
                if j in possibleCosts:
                    templist.append(j)
            
            #b=randint(38, mu-sigma-1)
            b=choice(templist)
            binstest.append(38)
            binstest.append(mu-1.3125*sigma)
            test.append(b)
            #Vet inte om denna ska vara kommenterad eller inte 
        elif i == 6:
            #up to highest we can buy from 
            templist=[]
            if budget-sum(test) > mu+sigma+1:
                for j in range(mu+sigma+1, budget-sum(test)):
                    if j in possibleCosts:
                        templist.append(j)
                        
                #b=randint(mu+sigma+1, budget-sum(test))
                if len(templist)>0:
                    b=choice(templist)
                else:
                    lastcost = [j for j in possibleCosts if j < budget-sum(test)]
                    b=choice(lastcost)
                test.append(b)
                binstest.append(mu+sigma+1)
                add= mu+sigma+1+sigma*0.3125

                while add < b:
                    binstest.append(add)
                    add+=sigma*0.3125
                binstest.append(b+1)
            else:
                for j in range(budget-sum(test)-12, budget-sum(test)):
                    if j in possibleCosts:
                        templist.append(j)
                #b=randint(budget-sum(test)-10,budget-sum(test))
                b=choice(templist)
                test.append(b)
                binstest.append((mu+1.3125*sigma))
                
        else:
            binstest.append((mu-sigma)+(i-1)*step)
            templist=[]
            for j in range((mu-sigma)+(i-1)*step, (mu-sigma-1)+(i)*step):
                if j in possibleCosts:
                    templist.append(j)
            a=1 
            while a <= idx :
                
                #b= randint((mu-sigma)+(i-1)*step, (mu-sigma-1)+(i)*step)
                b=choice(templist)
                test.append(b)
                a+=1
        
    #a = random.randint(75,85)
    # test = [75, 75, 75, 85,85, 65,65, 55, 95, 105, 45]
    teampoints=0
    for cost in test: 
        teampoints += choice(theList[cost])
    return sum(test), teampoints, test
#%% 
###DENNA KÖR MAN FÖR ATT FÅ FRAM NORMAL
theList = createTheList(1718)  
possibleCosts = createPossibleCosts(theList)    
budget = 950
allpoints=[]
allcosts=[]
for i in range(10):
    mu, sigma,step = calcMuSigmaStep(budget)
    tcost,tpoints, distr = generateRandomNormal(budget, mu, sigma, step, possibleCosts)  
    allpoints.append(tpoints)
    allcosts.append(tcost)
plotHistOfAllCostsAndPoints(allcosts, allpoints, budget, ' normal')  

print('Mean of 50 worst normal', sum(sorted(allpoints)[:50])/50) 
print('Mean all normal', sum(allpoints)/runs)
print('Mean of 50 best normal', sum(sorted(allpoints)[-50:])/50)


#%%
mu = round(budget/11)
sigma = round((mu-38)/1.645)
step=round(2*sigma/5)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma)*30)
plt.axvline(x=mu-sigma, color='r', ls='--')
plt.axvline(x=mu+sigma, color='r', ls='--')
plt.axvline(x=mu-1.645*sigma, color='b', ls='--')
plt.axvline(x=mu+1.645*sigma, color='b', ls='--')
# fit = stats.norm.pdf(h, np.mean(h), np.std(h))*2  #this is a fitting indeed
#plt.plot(h,fit,'-o')
#bins=[38, mu-1.3125*sigma,mu-sigma, mu-sigma+step,mu-sigma+2*step,mu+sigma-2*step+1,mu+sigma-step+1, mu+sigma+1, mu+1.3125*sigma+1, mu+1.625*sigma+1]
plt.hist(costs,bins=binstest)#, density=True)
plt.show()

#%%

def generateRandomLinearBins(theList, mean, possibleCosts, k, bins):
   # if bins[0] + k < 38:
    #    print('too small first bin')
   #     return 0,0,0
    totalPoints, totalCost, teamDistr = 0, 0, []
    for i,b in enumerate(bins):
        if i==11:
            break
    #    print('low, high: ' ,b, bins[i+1]-1)
        # Testing new approach, think it works and better 
        #lbin= b+1
        lbin = b
        hbin = bins[i+1]
        
        posCosts= [i for i in possibleCosts if i <=hbin and i >= lbin ]
        while len(posCosts) == 0:
              lbin -= 1 
              hbin += 1
              posCosts= [i for i in possibleCosts if i <=hbin and i >= lbin ]
        
        cost = choice(posCosts)
        templist = theList[cost]
        # THIS APPROACH WORKS
    #     lbin= b+1
    #     hbin = bins[i+1]
    #     cost = randint(lbin, hbin)
    #     while cost >= len(theList):
    #           cost = randint(lbin,hbin)
    # #    print('cost', cost)     
    #     templist = theList[cost]
    #     while (templist is None): 
    #          cost=randint(lbin,hbin)
    #          while cost >= len(theList):
    #               cost=randint(lbin,hbin) 
    #          print('cant find: ', cost)     
    #          templist = theList[cost]
        
        teamDistr.append(cost)    
        totalPoints += choice(templist) # add points
        totalCost += cost
    
    return totalPoints, totalCost, teamDistr

#%%

#RUN THIS FOR LINEAR 
budget = 900
lowerbudget =budget-50
allpoints=[]
allcosts=[]
alldistr=[]
while len(allcosts) < 1000:
    tpoints, tcost, distr = generateRandomLinearBins(theList, budget, possibleCosts)    
    if tcost < budget and tcost > lowerbudget:
        allpoints.append(tpoints)
        allcosts.append(tcost)
        alldistr.append(distr)
plotHistOfAllCostsAndPoints(allcosts, allpoints, budget, 'normal')  

print('Mean of 50 best', sum(sorted(allpoints)[-50:])/50)

#%%
plt.plot(range(11), alldistr[0], 'o')
plt.plot()
#%%
#Not the best... maybe not linear... 
def generateRandomLinearLine(theList, budget):
    n=11
    mean = int(budget/n)
    print('mean',mean)
    k = randint(1,2)
    print('k',k)
    m= mean-k*5
    print('m', m)
    
    x = np.linspace(0,10, 1000)
    y= k*x+m

    totalPoints, totalCost = 0, 0
    teamDistr = []
    for i in range(11):
        idx = randint(0,999)
        cost= round(y[idx])
        
        teamDistr.append(cost)    
        totalPoints += choice(templist) # add points
        totalCost += cost
    
    return totalPoints, totalCost, teamDistr

#%%
#Blir sådär...
budget=700
a,b,c = generateRandomLinearLine(theList, budget)
print('point', 'cost', 'distr',a,b,c)

#%%
 
def calcMuSigmaStep(budget):
    mu = round(budget/11)
    sigma = round((mu-38)/1.645)
    step=round(2*sigma/5)
    
    return mu, sigma, step

#%%
#Kör för att jämföra : 
    
#FOR NORMAL, MAYBE NOT START WITH 38 ? CHECK MORE  
seasons = [0] # 0 för Allsvenskan  

runs= 100000
n=11

for season in seasons:  
    print('------------------------SEASON', season, '------------------')
    theList = createTheList(season)  
    possibleCosts = createPossibleCosts(theList)    
    
    # n for normal, l for linear
    nallpoints, nallcosts, nalldistr=[], [],[]
    lallpoints, lallcosts, lalldistr=[], [],[]
    
    #Bästa k för varje budget enligt säsong 1617:
    budgets = [600, 650, 700, 750, 800, 850, 900, 950, 1000]
    bestk = [    1,   1,   1,   4,   6,   6,   8,   7,    6]
    #k = randint(1,10) #Trying with random k
    
    randomResults = pd.DataFrame(columns=['Budget', 'Linear/Normal', 'Mean cost', 'Mean points', 'Mean 50 best p', 'Mean 50 worst p', 'Ratio points/costs'])
    idx = 0 
    for j, budget in enumerate(budgets):
        lowerbudget= budget-20
        mean = budget/n
        
        print('---------------budget: ',budget, '-------------------')
        #for k in range(1,11): #Trying with fixed k
        k = bestk[j]
        
        lallpoints, lallcosts, lalldistr=[], [],[]
        lowbin= int(mean-k*5.5)
        bins= [lowbin + i*k for i in range(12)]
        if bins[0]+k<38:
            break
        
        while len(lallcosts) < runs: 
            ltpoints, ltcosts, ldistr = generateRandomLinearBins(theList, mean, possibleCosts, k, bins)    
            if ltcosts < budget and ltcosts > lowerbudget:
                lallpoints.append(ltpoints)
                lallcosts.append(ltcosts)
                lalldistr.append(ldistr)
        
        print('Mean of 50 worst linear', sum(sorted(lallpoints)[:50])/50) 
        print('Mean all linear', sum(lallpoints)/runs)
        print('Mean of 50 best linear', sum(sorted(lallpoints)[-50:])/50)             
          
        nallpoints, nallcosts, nalldistr=[], [],[]      
        while len(nallcosts) < runs: 
            mu, sigma, step = calcMuSigmaStep(budget)
            ntcosts, ntpoints, ndistr = generateRandomNormal(budget,mu,sigma,step, possibleCosts)  
            # kan skicka in mu, sigma, step för snabbhet
            if ntcosts < budget and ntcosts > lowerbudget:
                nallpoints.append(ntpoints)
                nallcosts.append(ntcosts)
                nalldistr.append(ndistr)
                    
        print('Mean of 50 worst normal', sum(sorted(nallpoints)[:50])/50) 
        print('Mean all normal', sum(nallpoints)/runs)
        print('Mean of 50 best normal', sum(sorted(nallpoints)[-50:])/50)
        
        plotHistOfAllCostsAndPoints(lallcosts, lallpoints, budget, ' Linear', season)      
        plotHistOfAllCostsAndPoints(nallcosts, nallpoints, budget, ' Normal', season)  

        randomResults.loc[idx] = ([budget, 'Linear: ' + str(k), int(sum(lallcosts)/runs), round(sum(lallpoints)/runs), round(sum(sorted(lallpoints)[-50:])/50), round(sum(sorted(lallpoints)[:50])/50), round(sum(lallpoints)/sum(lallcosts),3)])
        idx+=1
        randomResults.loc[idx] = ([budget, 'Normal' , round(sum(nallcosts)/runs), round(sum(nallpoints)/runs), round(sum(sorted(nallpoints)[-50:])/50), round(sum(sorted(nallpoints)[:50])/50), round(sum(nallpoints)/sum(nallcosts),3)])
        idx+=1
        
        #plot the different distributions
        plotDistr(lalldistr, 'Linear', budget, season)
        plotDistr(nalldistr, 'Normal', budget, season)
    
    randomResults.to_csv('results/pl/' + str(season) + '/generateRandom.csv')


#%%
#%%
def plotDistr(distr, title, budget, season):
    #plot the different distributions
    print('Plotting')
    flat_distr = [item for sublist in distr for item in sublist]

    plt.hist(flat_distr, range=(38,128), bins=bins)
    plt.title("Individual costs with distr: " + title+ ', Season: ' + str(season) + ',Budget: ' +str(budget) )
    plt.show()

    for i in range(1000):
        plt.plot(range(11), distr[i], 'o') 
    plt.title("Individual costs with distr: " + title+ ', Season: ' + str(season) + ',Budget: ' +str(budget) )
    plt.show()
  #%%
#plot the different distributions
print('Plotting')
flat_nalldistr = [item for sublist in nalldistr for item in sublist]
flat_lalldistr = [item for sublist in lalldistr for item in sublist]
#%%
plt.hist(flat_nalldistr, range=(38,128), bins=bins)
plt.title()
plt.show()
plt.hist(flat_lalldistr, range=(38,128), bins=bins)
plt.show()
#%%
for i in range(1000):
    plt.plot(range(11),lalldistr[i], 'o')
plt.show() 
for i in range(1000):
    plt.plot(range(11), nalldistr[i], 'o')
print('Done plotting')    
    
#%%
print('Mean all normal points', sum(nallpoints)/runs)
print('Mean all linear points', sum(lallpoints)/runs)
print('Mean of all costs linear', sum(lallcosts)/runs)
print('Mean of all costs normal', sum(nallcosts)/runs)
print(round(sum(nallpoints)/sum(nallcosts),3)) 
print(sum(lallpoints)/sum(lallcosts))   

#%%
randomResults.to_csv('results/pl/1617/generateRandom.csv')

# In[]

season = 1617
def createTheList(season):
    
    if season !=0:
   #     csv_file = "data/pl_csv/players_raw_" + str(season) + ".csv"
  #      playerspl = pd.read_csv(csv_file) 
 #       playerspl = playerspl.to_dict('index')
        playerspldata = getters.get_players_feature_pl("data/pl_csv/players_raw_", season)
        #gk, df, mf,fw = getters.get_diff_pos(playerspldata)
    
    if season==0:
#        csv_file='data/allsvenskan/players_raw.csv'
        playerspldata = getters.get_players_feature_pl('data/allsvenskan/players_raw', '')
    #allmfCost=[]
    #for m in mf.items():
    #    allmfCost.append(m[1]['now_cost'])

    #occurrences = collections.Counter(allmfCost)
    #print(sorted(occurrences.items()))    
    #plt.hist(allmfCost)

    sortIdxByCost = sorted(playerspldata, key=lambda k: (playerspldata[k]['now_cost']))

    test_dictionary = { i : playerspldata[idx] for idx, i in zip(sortIdxByCost, range(len(sortIdxByCost))) }
    highcost = test_dictionary[len(test_dictionary)-1]['now_cost'] 
    print('max',highcost) 

    #for idx in sortIdxByCost:
        #print(playerspldata[idx])
        
    print(playerspldata[sortIdxByCost[2]])
    
    value = -1
    theList = [None]*(highcost+1) #127 highest value for cost
    templist=[]
    for key in test_dictionary.values(): 
        if key['now_cost'] <= value:
            templist.append(key['total_points'])
        else: 
            theList[value] = templist  
            templist = []
            value = key['now_cost']
            templist.append(key['total_points'])
            if key['now_cost'] == highcost:
                theList[value] = templist    
    return theList

#%%
ASlist = createTheList(0)
randomResults.to_csv('results/as/generateRandom.csv')


#%%
def createPossibleCosts(theList):
    possibleCosts=[]
    for i,item in enumerate(theList):
        if item != None:
            possibleCosts.append(i)
    return possibleCosts    


#%%
season=2021
csv_file = "results/pl/" + str(season) + "/generateRandom.csv"
randomResult = pd.read_csv(csv_file)

costlin = randomResult['Mean cost'][::2]  
costnor = randomResult['Mean cost'][1::2]

plin = randomResult['Mean points'][::2]  
pnor = randomResult['Mean points'][1::2]

blin = randomResult['Mean 50 best p'][::2]  
bnor = randomResult['Mean 50 best p'][1::2]

wlin = randomResult['Mean 50 worst p'][::2]  
wnor = randomResult['Mean 50 worst p'][1::2]

X = np.arange(9)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, costlin, color = 'b', width = 0.25)
ax.bar(X + 0.25, costnor, color = 'g', width = 0.25)
ax.legend(labels=['Linear', 'Normal'])
realX= [x+0.125 for x in X ]
ax.set_xlabel('Budget')
ax.set_ylabel('Cost')
ax.set_title('Mean cost of team for each budget for season ' +str(season))
ax.set_xticks(realX,range(600,1050, 50))
plt.savefig('plots/randomGenerated/' + str(season) + 'cost.png', bbox_inches='tight')
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, plin, color = 'b', width = 0.25)
ax.bar(X + 0.25, pnor, color = 'g', width = 0.25)
ax.legend(labels=['Linear', 'Normal'])
realX= [x+0.125 for x in X ]
ax.set_xlabel('Budget')
ax.set_ylabel('Points')
ax.set_title('Mean points of team for each budget for season ' +str(season))
ax.set_xticks(realX,range(600,1050, 50))
plt.savefig('plots/randomGenerated/' + str(season) + 'points1.png', bbox_inches='tight')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, blin, color = 'b', width = 0.25)
ax.bar(X + 0.25, bnor, color = 'g', width = 0.25)
ax.legend(labels=['Linear', 'Normal'])
realX= [x+0.125 for x in X ]
ax.set_xlabel('Budget')
ax.set_ylabel('Points')
ax.set_title('Mean 50 best points for each budget for season ' +str(season))
ax.set_xticks(realX,range(600,1050, 50))
plt.savefig('plots/randomGenerated/' + str(season) + 'points2.png', bbox_inches='tight')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, wlin, color = 'b', width = 0.25)
ax.bar(X + 0.25, wnor, color = 'g', width = 0.25)
ax.legend(labels=['Linear', 'Normal'])
realX= [x+0.125 for x in X ]
ax.set_xlabel('Budget')
ax.set_ylabel('Points')
ax.set_title('Mean 50 worst points for each budget for season ' +str(season))
ax.set_xticks(realX,range(600,1050, 50))
plt.savefig('plots/randomGenerated/' + str(season) + 'points3.png', bbox_inches='tight')
