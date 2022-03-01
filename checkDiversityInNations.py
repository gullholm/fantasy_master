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


csv_file= "data/nationalities/pl/2021/players2021.csv" 
csv_file2 = "data/nationalities/pl/2021/players2021test.csv"  
nationalitiespl = pd.read_csv(csv_file)
nationalitiespltest = pd.read_csv(csv_file2, sep='\t')

total_players = [ast.literal_eval(nationalitiespltest['Players'][j])for j in range(len(nationalitiespltest))].sum()
nationalitiespltest['Percentage']= [nationalitiespltest['Players'][i]/total_players for i in range(len(nationalitiespltest))]

playerspernat= []
for j in range(len(nationalitiespl)):
    nation = nationalitiespl['List'][j].split(" ")
    playerspernat.append([nation[i] +' '+ nation[i+1] for i in range(0,int((len(nation)))-1,2)])

plpernat = [ast.literal_eval(nationalitiespltest['Names'][j]) for j in range(len(nationalitiespltest))] 
   

results = pd.read_csv('results/pl/2021/best.csv')

names = [ast.literal_eval(results['Name'][i]) for i in range(len(results))]

def checkNationality(namelist):
    
    for name in namelist:
        for nat in playerspernat: 
            for pl in nat:
                if pl==name: 
                    print(name)
                    break
        
        




#%%

checkNationality(names[0])
    
    
 
    
 
    