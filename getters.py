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

The page is now updated for the next season, 
uncomment if we want to download the current season

"""
#

# def get_data(serie = "allsvenskan", landskod = "se"):
#     """ Retrieve the fpl player data from the hard-coded url
#     """
#         ### Fixa så det kan bli PL också
#     response = requests.get("https://fantasy." + serie + "." + landskod + "/api/bootstrap-static/")
#     if response.status_code != 200:
#         raise Exception("Response was code " + str(response.status_code))
#     responseStr = response.text
#     data = json.loads(responseStr)
#     return data

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

def get_players_feature_pl(loc, season, list_feature = ['element_type', 'now_cost', 'total_points']):
    csv_file = str(loc) + str(season) + ".csv"
    playerspl = pd.read_csv(csv_file)
    playerspl = playerspl.to_dict('index')

    players_feature = {}

    for k,v in playerspl.items():
        case = {x: v.get(x) for x in list_feature}
        players_feature[v['id']] = case

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

"""
def get_cost_player(players, player_id):
    for (k,v) in players.items():
        if (k == player_id):
            return v.get("now_cost")
    return 0

def get_points_player(players, player_id):
    for (k,v) in players.items():
        if (k == player_id):
            return v.get("total_points")
    return 0

def get_cost_team(playersdata, team_id): # Team is list with id's

    team_cost = [get_cost_player(playersdata, ids) for ids in team_id]
    #team_cost = [cost_list[player_id] for player_id in team_id]
#    print(team_cost)
    team_cost.sort()
    return team_cost
"""

def get_full_name_pl(full_data, corr_id): # Get full name for a single player
    for player in full_data:
        if full_data[player]['id'] == corr_id:
            return full_data[player]['first_name'] + " " + full_data[player]['second_name']
    return 0

def get_full_name_team_pl(full_data, team_id): # Team is list with id's
    team_names = [get_full_name_pl(full_data, player_id) for player_id in team_id]
    return team_names

def get_teamName_pl(full_data, corr_id, season): # Get full name for a single player    
     listOfTeams, _ = seasonTeamsAndPlacement(season)
     for player in full_data:
        if (full_data[player]['id'] == corr_id):
            return(listOfTeams[full_data[player]['team']-1])
     return 0

def get_teamPos_pl(full_data, corr_id, season):
    _ , placementTeams = seasonTeamsAndPlacement(season)
    for player in full_data:
       if (full_data[player]['id'] == corr_id):
           return(placementTeams[full_data[player]['team']-1])

def get_teamName_and_pos_team_pl(data, team_id, season):
    teams = [get_teamName_pl(data, player_id, season) for player_id in team_id]
    teampos = [get_teamPos_pl(data, player_id, season) for player_id in team_id]
    return teams, teampos

def get_cost_player_pl(full_data, corr_id):

    for player in full_data:
        if (full_data[player]['id'] == corr_id):
            return full_data[player]['now_cost']
    return 0

def get_cost_team_pl(full_data, team_id): # Team is list with id's
    team_cost = [get_cost_player_pl(full_data, player_id) for player_id in team_id]
    return team_cost


def get_info_player_pl(info, full_data, corr_id):

        for player in full_data:
            if (full_data[player]['id'] == corr_id):
                return full_data[player][info]
        return 0

#info is the column in raw_data
def get_info_team_pl(info, full_data, team_id): # Team is list with id's
    info = [get_info_player_pl(info, full_data, player_id) for player_id in team_id]
    return info

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
    
    
    
def seasonTeamsAndPlacement(season):
    if season == 2021:
        listOfTeams = ["Arsenal", "Aston Villa", "Brighton & Hove Albion", 'Burnley', 'Chelsea',
 'Crystal Palace', 'Everton', 'Fulham','Leeds United','Leicester City','Liverpool','Manchester City',	
'Manchester United','Newcastle United', 'Sheffield United', 'Southampton',
 'Tottenham Hotspur', 'West Bromwich Albion', 'West Ham United','Wolverhampton Wanderers']
        placementTeams=[8,11,16,17,4,14,10,18,9,5,3,1,2,12,20,15,7,19,6,13]
    
    elif season == 1920:
        listOfTeams = ["Arsenal", "Aston Villa", 'Bournemouth', "Brighton & Hove Albion", 'Burnley', 'Chelsea',
     'Crystal Palace', 'Everton','Leicester City','Liverpool','Manchester City',	
    'Manchester United','Newcastle United', 'Norwich City', 'Sheffield United', 'Southampton',
     'Tottenham Hotspur', 'Watford', 'West Ham United','Wolverhampton Wanderers']
        placementTeams = [8,17,18,15,10,4,14,12,5,1,2,3,13,20,9,11,6,19,16,7]
    
    elif season == 1819:
           listOfTeams = ["Arsenal", 'Bournemouth', "Brighton & Hove Albion", 'Burnley', 'Cardiff City', 'Chelsea',
        'Crystal Palace', 'Everton', 'Fullham', 'Huddersfield Town', 'Leicester City','Liverpool','Manchester City',	
       'Manchester United','Newcastle United', 'Southampton',
        'Tottenham Hotspur', 'Watford', 'West Ham United','Wolverhampton Wanderers']
           placementTeams = [5,14,17,15,18,3,12,8,19,20,9,2,1,6,13,16,4,11,10,7]
   
    elif season == 1718:
           listOfTeams = ["Arsenal", 'Bournemouth', "Brighton & Hove Albion", 'Burnley', 'Chelsea',
        'Crystal Palace', 'Everton', 'Huddersfield Town', 'Leicester City','Liverpool','Manchester City',	
       'Manchester United','Newcastle United', 'Southampton', 'Stoke City', 'Swansey City',
        'Tottenham Hotspur', 'Watford', ' West Bromwich Albion', 'West Ham United']
           placementTeams = [6,12,15,7,5,11,8,16,9,4,1,2,10,17,19,18,3,14,20,13]


    elif season == 1617:
        listOfTeams=['Arsenal', 'Bournemouth', 'Burnley', 'Chelsea',
        'Crystal Palace', 'Everton', 'Hull City', 'Leicester City', 'Liverpool',
        'Manchester City' , 'Manchester United', 'Middlesbrough', 'Southampton',
        'Stoke City', 'Sunderland', 'Swansea City', 'Tottenham Hotspur', 'Watford',
        'West Bromwich', 'West Ham United']
        placementTeams=[5,9,16,1,14,7,18,12,4,3,6,19,8,13,20,15,2,17,10,11]
        
    return listOfTeams, placementTeams