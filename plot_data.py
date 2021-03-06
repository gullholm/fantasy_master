# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:50:35 2022

@author: jonat
"""

#Plot data for understanding and discussion

import matplotlib.pyplot as plt 
import pandas as pd
import getters
import calculations as calc
import numpy as np
import ast

# In[]
"""
Create different plots 
"""

# plot histogram of how many of each position
def plot_per_position(positions, title):
    
    x=["gk", "df", "mf", "fw" ]
    plt.plot(x, positions, 'o')
    plt.xlabel("Position")
    plt.ylabel("Amount")
    plt.title(title)
    ymin, ymax = [0, 310]
    plt.ylim(ymin,ymax)    
    plt.show()


# plot hist of points
def plot_hist_of_points(feature_data, title):
    pointsList= calc.createPointsList(feature_data)
    plt.hist(pointsList)
    plt.xlabel("Points")
    plt.ylabel("Amount")
    xmin, xmax, ymin, ymax = [0, 320, 0, 350]
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title(title)
    plt.show()

# plot hist of costs
def plot_hist_of_costs(feature_data, title):
    costList= calc.createCostList(feature_data)
    #npscostlist = [np.log(x) for x in costList if x > 0]
    plt.hist(costList)
    #plt.hist(npscostlist)
    plt.xlabel("Costs")
    plt.ylabel("Amount")
    xmin, xmax, ymin, ymax = [35, 140, 0, 300]
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title(title)
    plt.show()
    
def plotIndividualCosts(feature_data, title, degree, line=False):

    x = list(range(1,12,1))
    plt.xlabel("Player")
    plt.ylabel("Cost")
    for i in range(len(feature_data)):
        y = ast.literal_eval(feature_data[i]['Individual costs'])
        plt.plot(x,y, 'o', label = '< %s'%(500+i*50))
        if line:
            poly= np.polyfit(x,y,degree)
            plt.plot(x, np.polyval(poly,x))
    plt.legend()
    plt.title(title)
    plt.show()   

#%%
"""
Get the data
"""
#PL
seasons=[1617,1718,1819,1920,2021]

for season in seasons:
#As
    if season == 0:
        title = "Allsvenskan" 
        
        data = getters.get_data()
        playersdata = getters.get_players_feature(data)
        gk, df, mf, fw = getters.get_diff_pos(playersdata)
        
        csv_results = "results/as/best.csv"
        results = pd.read_csv(csv_results).to_dict('index') 
    
    else:
        title = "Premier league season " + str(season)
        
        csv_file = "data/pl_csv/players_raw_" + str(season) + ".csv"
        playerspl = pd.read_csv(csv_file).to_dict('index') 
        playersdata = getters.get_players_feature_pl("data/pl_csv/players_raw_", season)
        gk, df, mf, fw = getters.get_diff_pos(playersdata)  
        
        csv_results = "results/pl/" + str(season) + "/best.csv"
        results = pd.read_csv(csv_results).to_dict('index') 

        
    
    positions= [len(gk), len(df), len(mf), len(fw)]
        
    plot_per_position(positions, title)
    plot_hist_of_costs(playersdata, title)
    plot_hist_of_points(playersdata, title)
    plotIndividualCosts(results, title, degree=1, line=False)


           

# In[]


best_costs = results[7]['Individual costs']
ind_costs = [50,53,55,55,56,57,66,96,113,119,129]
x = list(range(1,12,1))
y = ind_costs

plt.xlabel("Player")
plt.ylabel("Cost")
plt.legend()
plt.show()  
# In[9]
#Theory , not done
# minimum salary, in our case 37?
# total budget, in our case 700
# C = 700/11 ~63.64, a = 37

#spanning the range from a to 2C-a

# maybe total budget should be the total cost of the best team 
# maybe minimum salary, "a", should be the cheapest in the best team

for idx in range(len(y)):

    poly= np.polyfit(x,y[idx],1)
    plt.plot(x, np.polyval(poly,x))
    
    theory_m = y[idx][0]
    theory_expensive = round(sum(y[idx])*2/11-theory_m)  
    theory_k = np.float64(theory_expensive-theory_m)/11
    plt.plot(x, theory_k*x+theory_m)
    
    total_diff = 0
    for i in range(1,12):
        first = k*i+m
        second =  theory_k*i + theory_m
        diff = np.abs(first-second)
        total_diff += diff
    
    mean_diff =  total_diff/11
    print(mean_diff)
    
sum_best_costs = list(map(sum, best_costs))
sum_best_points = list(map(sum, best_points))


 # In[] 
import RandomGenreateTeams
eason=1819
cleaned = RandomGenreateTeams.cleanAllPositions(season)  
# In[]
titlenames = ["df3", 'df4','df5', 'mf3', 'mf4', 'mf5', 'fw1', 'fw2', 'fw3', 'gk']
for j in range(len(cleaned)):
    totcost=[]
    for i in range(len(cleaned[j])):
        totcost.append(cleaned[j].iloc[i]['now_cost'])
    plt.hist(totcost)
    plt.xlabel("Cost")
    plt.ylabel
    plt.title("Individuella kostnader efter cleaned, "
              "innan cleaned som formation: " + titlenames[j] + 
              " season: " + str(season))
    plt.show()
 
# In[]
    
def getResultsPerSeason(season):
    if season == 0: 
        csv_results = "results/as/best.csv"
        results = pd.read_csv(csv_results).to_dict('index') 
    else: 
        csv_results = "results/pl/" + str(season) + "/best.csv"
        results = pd.read_csv(csv_results).to_dict('index') 
        
    return results 
 
# In[]        

# Mean of all PL data

seasons=[0, 1617, 1718, 1819, 1920, 2021]

allResults = []
for season in seasons: 
    allResults.append(getResultsPerSeason(season))        

meanIndCost = []
for j in range(len(allResults[1])): 
    for i in range(1,6):
        if i == 1:
            meanIndCost.append(ast.literal_eval(allResults[i][j]['Individual costs']))
        else: 
            lista = meanIndCost[j]
            listb = ast.literal_eval(allResults[i][j]['Individual costs'])
            meanIndCost[j] = [lista[i] + listb[i] for i in range(11)]
            
for k in range(11):    
    meanIndCost[k] = [x/5 for x in meanIndCost[k]]   
    
degree=3

x = list(range(1,12,1))
plt.xlabel("Player")
plt.ylabel("Cost")
line=True
for i in range(len(meanIndCost)):
    y = (meanIndCost[i])
    plt.plot(x,y, 'o', label = '< %s'%(500+i*50))
    if line:
        poly= np.polyfit(x,y,degree)
        plt.plot(x, np.polyval(poly,x))
plt.legend()
plt.title("test")
plt.show()
    
    
    
#%%
import getters
generic = lambda x: ast.literal_eval(x)


seasons=[0, 1617, 1718, 1819, 1920, 2021]

allResults = []
for season in seasons: 
    allResults.append(getResultsPerSeason(season))     

def getIdFromBestTeam(playerspl, season):
    seasons=[0, 1617, 1718, 1819, 1920, 2021]
    i = seasons.index(season)
    indexes=[]
    plDataFrame = pd.DataFrame()
    
    for j in range(len(allResults[1])):
        indexes = ast.literal_eval(allResults[i][j]['Id'])
        best_names = getters.get_full_name_team_pl(playerspl, indexes)
        teamNames, teamPositions = getters.get_teamName_and_pos_team_pl(playerspl, indexes, season)
        inDreamteam = getters.get_info_team_pl('in_dreamteam', playerspl, indexes)
        percentage = getters.get_info_team_pl('selected_by_percent', playerspl, indexes)
        monthsInDreamteam= getters.get_info_team_pl('dreamteam_count',playerspl, indexes)
        goals = getters.get_info_team_pl('goals_scored', playerspl, indexes)
        assists = getters.get_info_team_pl('assists', playerspl, indexes)
        bonus = getters.get_info_team_pl('bonus', playerspl, indexes)
        clean_sheets = getters.get_info_team_pl('clean_sheets', playerspl, indexes)
        goals_conceded = getters.get_info_team_pl('goals_conceded', playerspl, indexes)
        minutes = getters.get_info_team_pl('minutes', playerspl, indexes)
        saves = getters.get_info_team_pl('saves', playerspl, indexes)
        ycard = getters.get_info_team_pl('yellow_cards', playerspl, indexes)
        rcard = getters.get_info_team_pl('red_cards', playerspl, indexes)

        allResults[i][j]['Name'] = best_names
        allResults[i][j]['Team'] = teamNames
        allResults[i][j]['Team position'] = teamPositions
        allResults[i][j]["Dreamteam"] = inDreamteam 
        allResults[i][j]["Amount in Dreamteam"] = sum(inDreamteam)
        allResults[i][j]["Months in Dreamteam"] = monthsInDreamteam        
        allResults[i][j]["Selected by percentage"] = percentage
        allResults[i][j]['Goals'] = goals
        allResults[i][j]['Assists'] = assists
        allResults[i][j]['Bonus points'] = bonus 
        allResults[i][j]['Clean Sheets'] = clean_sheets
        allResults[i][j]['Goals conceded'] = goals_conceded
        allResults[i][j]['Total minutes']  = minutes
        allResults[i][j]['Total saves'] = saves
        allResults[i][j]['Yellow cards'] = ycard
        allResults[i][j]['Red cards'] = rcard
        
        
        data = pd.DataFrame.from_dict(allResults[i][j], orient='index').transpose()    
        plDataFrame = plDataFrame.append(data)
        
    plDataFrame.to_csv("results/pl/" + str(season) + "/best.csv",index = False)
        
    return plDataFrame

#%%


seasons=[1617, 1718, 1819, 1920, 2021]
for season in seasons: 
    conv = {'indexes': generic}
    csv_file = "data/pl_csv/players_raw_" + str(season) + ".csv"
    playerspl = pd.read_csv(csv_file).to_dict('index')
    
    test = getIdFromBestTeam(playerspl, season)   
    
#%%
# Save results per budget, not per season

seasons=[1617, 1718, 1819, 1920, 2021]


allResults = []
for season in seasons: 
    allResults.append(getResultsPerSeason(season))       


for j in range(len(allResults[0])):
    budgetFrame = pd.DataFrame()
    budget = allResults[0][j]['Budget']
    for i in range(5):
        data = pd.DataFrame.from_dict(allResults[i][j], orient='index').transpose()    
        budgetFrame = budgetFrame.append(data) 
        budgetFrame = budgetFrame.drop('Budget', axis=1)
    
    budgetFrame.to_csv("results/pl/budget/" + str(budget) + ".csv",index = False)



#%%

def getResultsPerBudget(budget): 
    csv_results = "results/pl/budget/" + str(budget) +".csv"
    results = pd.read_csv(csv_results).to_dict('index')    
    return results

budgets = list(range(500,1050, 50))
budgetResults =[]
for budget in budgets:
    budgetResults.append(getResultsPerBudget(budget))    

sortedCosts = []  
for j in range(len(budgetResults)):
    print(j)
    templist = []
    for i in range(len(budgetResults[j])):
    
        templist.append(list(ast.literal_eval(budgetResults[j][i]['Sorted individual costs'])))

    sortedCosts.append(templist)

#%%
#root mean square error
def rmse(actual, predicted):
    return np.sqrt(np.square(np.subtract(np.array(actual), np.array(predicted))).mean())

# root mean square percentage error
def rmspe(actual, predicted):
    return np.sqrt(np.mean(np.square((actual - predicted) / actual))) * 100


def testIfLinear(data, budget):
    x=range(1,12)
    print(budget)
    for degree in range(1,4):
        poly= np.polyfit(x,data,degree)
        ypred = np.polyval(poly,x)
        plt.plot(x, ypred)

    plt.title("mean for: " + str(budget))
    plt.xlabel("Player")
    plt.ylabel("Normalized cost")
    plt.plot(x, data, 'o')
    plt.legend(["Linear", "Quadtratic", "Third degree polynomial", "Data"])
    plt.show()
     
#%%

import numpy as np
# calculate mean of multiple lists
meanCostsPerBudget =[]
for i in range(len(sortedCosts)):
                     
    arrays = [np.array(x) for x in sortedCosts[i]]
    meanofLists = [np.mean(k) for k in zip(*arrays)]    
    meanCostsPerBudget.append(meanofLists)

#%%
i=0
for bud in meanCostsPerBudget:
    testIfLinear(bud, 500+i) 
    i += 50             

#%%

for bud in sortedCosts[0]:
    testIfLinear(bud, 500)

#%% 

# Plot some mean values for each budget :
    
templistTwo = []

for j in range(len(budgetResults)):
    templistOne = []
    for i in range(len(budgetResults[j])):
    
        templistOne.append(budgetResults[j][i]['Best total points'])

    templistTwo.append(templistOne)
    
for i in range(len(sortedCosts)):
                           
    budgetMeanPoints = [np.mean(x) for x in templistTwo]
x=range(500, 1050, 50)
plt.xlabel("Budget")
plt.ylabel("Points")
plt.title("Mean points for different budgets for seasons from 16/17 to 20/21")
plt.plot(x, budgetMeanPoints, 'o')

#%%     
templistTwo = []

for j in range(len(budgetResults)):
    templistOne = []
    for i in range(len(budgetResults[j])):
    
        templistOne.append(budgetResults[j][i]['Amount in Dreamteam'])

    templistTwo.append(templistOne)
    
for i in range(len(sortedCosts)):
                           
    budgetMeanAmounts = [np.mean(x) for x in templistTwo]
x=range(500, 1050, 50)
plt.xlabel("Budget")
plt.ylabel("Amount")
plt.title("Mean amounts in dreamteam for different budgets for seasons from 16/17 to 20/21")
plt.plot(x, budgetMeanAmounts, 'o')


#%%
x=list(range(500, 1050, 50))

for i in range(len(sortedCosts)):
                           
    meanmean = [np.mean(x) for x in meanCostsPerBudget]    


for i in range(len(x)):
    plt.scatter([x[i]]*len(meanCostsPerBudget[i]),meanCostsPerBudget[i])
plt.plot(x, meanmean, 'o', markersize=12, color='black', label="Mean")
plt.legend()
plt.xlabel("Budget")
plt.ylabel("Costs")
plt.title("Mean costs for different budgets for seasons from 16/17 to 20/21")

#%%
from collections import Counter
templistTwo = []

for j in range(len(budgetResults)):
    templistOne = []
    for i in range(len(budgetResults[j])):
    
        templistOne.extend(ast.literal_eval(budgetResults[j][i]['Name']))

    templistTwo.append(templistOne)

x=list(range(500, 1050, 50))

occurencesPlayers = []
for i in range(len(templistTwo)):
    a = pd.Series(templistTwo[i]).value_counts()
    b = Counter(a)
    c = list(b.values())
    c.reverse()
    occurencesPlayers.append(c)
    
x=list(range(500, 1050, 50))
for i in range(len(x)):
    #plt.scatter([x[i]]*len(occurencesPlayers[i]),occurencesPlayers[i])
    plt.plot(list(range(1,len(occurencesPlayers[i])+1)), occurencesPlayers[i], 'o')


plt.xlabel("Occurences")
plt.ylabel("Amount")
plt.title("Occurences of players for different budgets for seasons from 16/17 to 20/21")

#%%

def plotDifferentResults(data, info, ylab):
    
    dataPart= []
    
    for j in range(len(data)):
        tempOne = []
        for i in range(len(data[j])):
        
            tempOne.append(ast.literal_eval(data[j][i][info]))
    
        dataPart.append(tempOne)
    
    hej =[]
    for i in range(len(sortedCosts)):
                         
        arrays = [sorted(np.array(x)) for x in dataPart[i]]
        meanofLists = [np.mean(k) for k in zip(*arrays)]    
        hej.append(meanofLists)    
    
    for i in range(len(sortedCosts)):
                               
        hejMean = [np.mean(x) for x in hej]
    
    x=list(range(500, 1050, 50))
    for i in range(len(x)):
        plt.scatter([x[i]]*len(hej[i]), hej[i])
    
    plt.plot(x, hejMean, 'o', markersize=12, color='black', label="Mean")
    plt.legend()
    plt.xlabel("Budget")
    plt.xticks(range(500, 1050, 50))
    plt.ylabel(ylab)
    plt.title(info)
    plt.show()
    
#%%    
plotDifferentResults(budgetResults, 'Months in Dreamteam', 'Months')  
plotDifferentResults(budgetResults, 'Selected by percentage', 'Percentage[%]') 
plotDifferentResults(budgetResults, 'Team position', 'Position')
plotDifferentResults(budgetResults, 'Yellow cards', 'Amount of yellow cards')
plotDifferentResults(budgetResults, 'Red cards', 'Amount of red cards')
plotDifferentResults(budgetResults, 'Goals', 'Amount of goals')
plotDifferentResults(budgetResults, 'Goals conceded', 'Amount of conceded goals')
plotDifferentResults(budgetResults, 'Bonus points', 'Amount of bonus points')
plotDifferentResults(budgetResults, 'Assists', 'Amount of assists')
plotDifferentResults(budgetResults, 'Total minutes', 'Amount of minutes')
plotDifferentResults(budgetResults, 'Clean Sheets', 'Amount of clean sheets')
   

#%%

allres = getResultsPerSeason('all') 
for i in range(11):  
    plt.plot(list(range(1,12)),ast.literal_eval(allres[i]['Sorted individual costs']), 'o') 
