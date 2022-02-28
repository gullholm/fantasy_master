import parsers
import pandas as pd
import os

def remove_expansive(loc = "data/pl_csv/players_raw_", season = 1617, dest = "data/pl_csv_no_exp/", prop = k):
    full_loc = loc + str(season) + ".csv"
    full_data = pd.read_csv(full_loc)
    full_data.sort_values(by = "now_cost", ascending = False)
    full_data.drop(index = range(round(len(full_data)*k)), axis = 0, inplace = True)
    os.makedirs(dest, exist_ok = True)
    full_dest = dest + str(season) + ".csv"
    full_data.to_csv(full_dest)
    