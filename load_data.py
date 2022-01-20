"""
Created on Thu Jan 20 14:06:26 2022

@author: josef
"""

import pandas as pd
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