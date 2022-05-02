# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 14:46:29 2022

@author: jonat
"""
#Check diversity

# link : https://fbref.com/en/comps/9/10728/nations/2020-2021-Premier-League-Nationalities
# Provided by <a href="https://www.sports-reference.com/sharing.html?utm_source=direct&utm_medium=Share&utm_campaign=ShareTool">FBref.com</a>: <a href="https://fbref.com/en/comps/9/10728/nations/2020-2021-Premier-League-Nationalities?sr&utm_source=direct&utm_medium=Share&utm_campaign=ShareTool#nations">View Original Table</a><br>Generated 2/28/2022.


import pandas as pd
import ast
from collections import Counter


csv_file2 = "data/nationalities/pl/nationalities.csv"  
nationalitiespl = pd.read_csv(csv_file2, sep='\t')

plpernat = [nationalitiespl['Names'][j].split(', ') for j in range(len(nationalitiespl))] 
   
def checkNationality(namelist, playerspernat):
    nations= []
    for name in namelist:
        for i, nat in enumerate(playerspernat): 
            for pl in nat:
                if pl==name: 
                    nations.append(nationalitiespl['Nation'][i])
                    break
    return nations

#%%
import matplotlib.pyplot as plt
seasons = [1617,1718,1819,1920,2021]

for season in seasons:
    print(season)
    csv_file = 'results/pl/' + str(season) +'/best.csv'
    results = pd.read_csv(csv_file)
    names = [ast.literal_eval(results['Name'][i]) for i in range(len(results))]

    csv_file2 = "data/nationalities/pl/" + str(season) + ".csv"  
    statsnations = pd.read_csv(csv_file2)   
    totLeague  = sum([nr for nr in statsnations['# Players']])
    statsnations['Percentage'] = [round(nr/totLeague,3) for nr in statsnations['# Players']]

    allData = pd.DataFrame(columns = ['Budget', 'Nation', 'Count', 'Team percentage', 'League percentage', 'Ratio', 'Amount as dist'])
    budget = 500
    for i in range(len(names)):

        nations = checkNationality(names[i], plpernat)
    
        natPlayers = pd.DataFrame.from_dict(Counter(nations), orient='index').reset_index()
        natPlayers =natPlayers.rename(columns={'index':'Nation', 0:'Count'})
        natPlayers['Team percentage'] = [round(temp/11,3) for temp in natPlayers['Count']]
    
        lPercentage = []
        for i in range(len(natPlayers)):
            idx = [nat for nat in statsnations['Nation']].index(natPlayers.iloc[i]['Nation'])
            lPercentage.append(statsnations['Percentage'][idx])
    
        natPlayers['League percentage'] = lPercentage
        natPlayers['Ratio'] = round(natPlayers['Team percentage']/natPlayers['League percentage'],3)
        natPlayers['Amount as dist'] = [round(lp*11,3) for lp in natPlayers['League percentage']]
        
        temp = [None]*7
        for i, (columnName, columnData) in zip(range(7),natPlayers.iteritems()):
            temp[i+1] = list(columnData)

        temp[0] = budget
        allData_length = len(allData)
        allData.loc[allData_length] = temp
        
        budget+=50


    allData.to_csv("results/pl/" + str(season) +"/nationality_diversity.csv")
    
    
   
#%%

seasons = [1617,1718,1819,1920,2021]

for season in seasons: 
    csv_file = "data/nationalities/pl/" + str(season) + ".csv"  
    statsnations = pd.read_csv(csv_file)   
    totLeague  = sum([nr for nr in statsnations['# Players']])
    statsnations['Percentage'] = [round(nr/totLeague,3) for nr in statsnations['# Players']]
    
    plt.bar(statsnations['Nation'][:10],statsnations['Percentage'][:10]*100)
    plt.xticks(rotation='vertical')
    plt.ylabel('Percent of league')
    plt.title('Top 10 nations in season: ' + str(season))
    plt.show()

#%%

seasons = [1617,1718,1819,1920,2021]

for season in seasons: 
    csv_file = "results/pl/" + str(season) + "/nationality_diversity.csv"  
    res_nations = pd.read_csv(csv_file)   
    nations = [ast.literal_eval(nat) for nat in res_nations['Nation']]
    teamperc = [ast.literal_eval(per) for per in res_nations['Team percentage']]           
    plt.bar(nations[0], teamperc[0])
    plt.ylabel('Percent of league')
    plt.title('Top 10 nations in season: ' + str(season))
    plt.show()