"""
Created on Thu Jan 20 14:06:26 2022

@author: josef
"""

import requests
import json

"""
Get all data for the current season in allsvenskan
"""

def get_data():
    """ Retrieve the fpl player data from the hard-coded url
    """
        ### Fixa så det kan bli PL också
    response = requests.get("https://fantasy.allsvenskan.se/api/bootstrap-static/")
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
    

"""
Split players by position

"""

def get_diff_pos(players_data): 
    
    goalkeepers = {k:v for (k,v) in players_data.items() if v['element_type']==1}
    defenders = {k:v for (k,v) in players_data.items() if v['element_type']==2}
    midfielders = {k:v for (k,v) in players_data.items() if v['element_type']==3}
    forwards = {k:v for (k,v) in players_data.items() if v['element_type']==4}
    
    return goalkeepers, defenders, midfielders, forwards

def get_full_name(full_data, corr_id): # Get full name for a single player
    players = full_data["elements"] 
    
    for player in players:
        if (player['id'] == corr_id):
            return player['first_name'] + " " + player['second_name']
    return 0