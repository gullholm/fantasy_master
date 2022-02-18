"""
Created on Thu Jan 20 14:06:26 2022

@author: josef
"""

import requests
import json
import os
import csv
import pandas as pd
import ast

"""
Get all data for the current season in allsvenskan
"""

def get_data(serie = "allsvenskan", landskod = "se"):
    """ Retrieve the fpl player data from the hard-coded url
    """
        ### Fixa så det kan bli PL också
    response = requests.get("https://fantasy." + serie + "." + landskod + "/api/bootstrap-static/")
    if response.status_code != 200:
        raise Exception("Response was code " + str(response.status_code))
    responseStr = response.text
    data = json.loads(responseStr)
    return data

"""
Get given feature for all players for the current season
"""

def get_players_feature(full_data, list_feature = ['element_type', 'now_cost', 'total_points']):
    players_list = full_data["elements"]
    players_feature = {}
    for player in players_list:
        case = {x: player[x] for x in list_feature}
        players_feature[player['id']] = case
    return players_feature  


def get_players_feature_pl(full_data, list_feature = ['element_type', 'now_cost', 'total_points']):
    players_feature = {}
    for i in range(len(full_data)):
        case = {x: full_data[i][x] for x in list_feature}
        players_feature[full_data[i]['id']] = case
    return players_feature      
    

"""
Split players by position

"""

def get_diff_pos(players_data): 
    
    goalkeepers = {k:v for (k,v) in players_data.items() if v['element_type']==1}
    defenders = {k:v for (k,v) in players_data.items() if v['element_type']==2}
    midfielders = {k:v for (k,v) in players_data.items() if v['element_type']==3}
    forwards = {k:v for (k,v) in players_data.items() if v['element_type']==4}
    
    return goalkeepers, defenders, midfielders, forwards

"""
 Parsers
"""


def extract_stat_names(dict_of_stats):
    """ Extracts all the names of the statistics
    Args:
        dict_of_stats (dict): Dictionary containing key-alue pair of stats
    """
    stat_names = []
    for key, val in dict_of_stats.items():
        stat_names += [key]
    return stat_names

def parse_players(list_of_players, base_filename):
    stat_names = extract_stat_names(list_of_players[0])
    filename = base_filename + 'players_raw.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    f = open(filename, 'w+', encoding='utf8', newline='')
    w = csv.DictWriter(f, sorted(stat_names))
    w.writeheader()
    for player in list_of_players:
            w.writerow({k:str(v).encode('utf-8').decode('utf-8') for k, v in player.items()})
            
def clean_players(filename, base_filename):
    """ Creates a file with only important data columns for each player
    Args:
        filename (str): Name of the file that contains the full data for each player
    """
    headers = ['first_name', 'second_name', 'goals_scored', 'assists', 'total_points', 'minutes', 'goals_conceded', 'clean_sheets', 'red_cards', 'yellow_cards', 'selected_by_percent', 'now_cost', 'element_type']
    fin = open(filename, 'r+', encoding='utf-8')
    outname = base_filename + 'cleaned_players.csv'
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    fout = open(outname, 'w+', encoding='utf-8', newline='')
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, headers, extrasaction='ignore')
    writer.writeheader()
    for line in reader:
        if line['element_type'] == '1':
            line['element_type'] = 'GK'
        elif line['element_type'] == '2':
            line['element_type'] = 'DEF'
        elif line['element_type'] == '3':
            line['element_type'] = 'MID'
        elif line['element_type'] == '4':
            line['element_type'] = 'FWD'
        else:
            print("Oh boy")
        writer.writerow(line)
        
    
def get_full_name(full_data, corr_id): # Get full name for a single player
    players = full_data["elements"] 
    
    for player in players:
        if (player['id'] == corr_id):
            return player['first_name'] + " " + player['second_name']
    return 0

def get_full_name_team(full_data, team_id): # Team is list with id's
    team_names = [get_full_name(full_data, player_id) for player_id in team_id]
    return team_names

def get_cost_player(full_data, corr_id):
    players = full_data["elements"] 
    
    for player in players:
        if (player['id'] == corr_id):
            return player['now_cost']
    return 0

def get_cost_team(full_data, team_id): # Team is list with id's
    team_cost = [get_cost_player(full_data, player_id) for player_id in team_id]
    return team_cost

# PL

def get_full_name_pl(full_data, corr_id): # Get full name for a single player
    
    for player in full_data:
        #print(player)
        if full_data[player]['id'] == corr_id:
            return full_data[player]['first_name'] + " " + full_data[player]['second_name']
    return 0

def get_full_name_team_pl(full_data, team_id): # Team is list with id's
    team_names = [get_full_name_pl(full_data, player_id) for player_id in team_id]
    return team_names

def get_teamName_pl(full_data, corr_id): # Get full name for a single player
    #print(corr_id)    
    for (k,v) in full_data.items():
        print(k)
        if (full_data[k]['id'] == corr_id):
            #print('hi')
            return full_data[k]['team']
        return 0

def get_teamName_team_pl(full_data, team_id):
    print('hi22')
    teams = [get_teamName_pl(full_data, player_id) for player_id in team_id]
    return teams

def get_cost_player_pl(full_data, corr_id):

    for player in full_data:
        if (full_data[player]['id'] == corr_id):
            return full_data[player]['now_cost']
    return 0

def get_cost_team_pl(full_data, team_id): # Team is list with id's
    team_cost = [get_cost_player_pl(full_data, player_id) for player_id in team_id]
    return team_cost



def get_cleaned_combs(base = "data_cleaned", files = ["gk", "df", "mf", "fw"]):
    # create empty list
    dataframes_list = []
    generic = lambda x: ast.literal_eval(x)
    conv = {'indexes': generic}
 
    # append datasets into teh list
    for f in files:
        temp_df = pd.read_csv(base + "/" + f + ".csv", converters = conv)
        if (f == "gk"): #Convert so also gk has indexes as lists
            temp_df['indexes'] = temp_df['indexes'].apply(lambda x: [x]) 
        dataframes_list.append(temp_df)
        
    return dataframes_list
    
    
    
