import parsers
import pandas as pd
import os

def remove_expansive(loc = "data/pl_csv/players_raw_", season = 1617, dest = "data/pl_csv/", k = 0.1):
    full_loc = loc + str(season) + ".csv"
    full_data = pd.read_csv(full_loc)
    full_data.sort_values(by = "now_cost", ascending = False,inplace = True)
    full_data.reset_index(inplace = True)
    full_data.drop(index = range(round(len(full_data)*k)), axis = 0, inplace = True)
    full_data.reset_index(inplace = True, drop = True)
    lgt = range(len(full_data))
    for i in lgt:
  #      print(i)
        full_data.at[i, "id"] = i+1
    os.makedirs(dest, exist_ok = True)
  #  print(full_data.id)
    print(lgt)

    full_dest = dest +"players_noexp_" + str(k) + "_" + str(season) + ".csv"
    full_data.to_csv(full_dest, index = False)


