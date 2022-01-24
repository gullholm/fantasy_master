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
    
def get_diff_pos(players_data):
    
    goalkeepers = [x for x in players_data if x['element_type']==1]
    defenders = [x for x in players_data if x['element_type']==1]
    midfielders = [x for x in players_data if x['element_type']==1]
    forwards = [x for x in players_data if x['element_type']==1]
    
    return goalkeepers, defenders, midfielders, forwards