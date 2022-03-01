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
                    #print(str(name) + ' is from ' +str(nationalitiespl['Nation'][i]))
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

    for i in range(len(names)):
        print('--------------------------------------')
        nations = checkNationality(names[i], plpernat)
        #print(len(nations))
        print(Counter(nations))

 
#%%
csv_file = "data/nationalities/pl/2021.csv"  
statsnations = pd.read_csv(csv_file)
#tot_play = [int(nationalitiespltest['Players'][j]) for j in range(len(nationalitiespltest))]
#total_players= sum(tot_play)
#nationalitiespltest['Percentage']= [nationalitiespltest['Players'][i]/total_players for i in range(len(nationalitiespltest))]
    
 
    