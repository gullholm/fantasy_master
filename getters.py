"""
Created on Thu Jan 20 14:06:26 2022

@author: josef
"""

import requests
import json


def get_data(league = "allsvenskan"):
    """ Retrieve the fpl player data from the hard-coded url
    """
        ### Fixa så det kan bli PL också
    response = requests.get("https://fantasy." + league + ".se/api/bootstrap-static/")
    if response.status_code != 200:
        raise Exception("Response was code " + str(response.status_code))
    responseStr = response.text
    data = json.loads(responseStr)
    return data
ssss = get_data()
response = requests.get("https://fantasy.allsvenskan.se/api/element-summary/523/")
data_pl = json.loads(response.text)

def get_players_feature(full_data, list_feature = ['id', 'element_type', 'now_cost', 'total_points']):
    players_list = full_data["elements"]
    players_feature = []
    for player in players_list:
        case = {x: player[x] for x in list_feature}
        players_feature.append(case)
    return players_feature        
    
    