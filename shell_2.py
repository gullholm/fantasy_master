import cleaners
import parsers

"""
CHOOSE LEAGUE & YEAR:
"""
league = "allsvenskan"
year = 2021

"""
^^^^^^^^^^
"""





all_parts_but_goalie = cleaners.all_forms_as_df_cleaned(league = league)[1:]
formations = [[3,4,5],[3,4,5],[1,2,3]]
form_name = ["df", "mf", "fw"]


for part, df, pos in zip(formations, all_parts_but_goalie, form_name):
    for p in part:
        all_cleaned = cleaners.run_all_cleans(df, p)
        combs = parsers.create_all_combs_from_cleaned_df(all_cleaned, p)[0]
        combs.to_csv("data_cleaned/as/" + pos + "/" + str(p) + ".csv",index = False)



def calc_full_teams(all_combs):
    all_combs[0]['indexes'] = all_combs[0].index
    all_combs[0]['indexes'] = all_combs[0]['indexes'].apply(lambda x: [x])

    all_points = calc.calc_from_combs(all_combs, "total_points")
    all_costs = calc.calc_from_combs(all_combs, "now_cost" )

    points_full = parsers.parse_formations_points_or_cost(all_points)
    costs_full = parsers.parse_formations_points_or_cost(all_costs)
    
    it = np.nditer(points_full, flags=['multi_index'])
    bit = np.nditer(costs_full, flags = ['multi_index'])
    sep_ids  = [combs['indexes'].values.tolist() for combs in reversed(all_combs)]
    costs_total, points_total, indexes = [], [], []

    for x, y in zip(it,bit):
        points_total.append(x)
        costs_total.append(y)
        reg_list = [x[it.multi_index[i]] for (i,x) in enumerate(sep_ids)]
        indexes.append([item for sublist in reg_list for item in sublist])
        
    full_teams_df = pd.DataFrame({"cost": costs_total, "points_total": points_total, "indexes": indexes})
    return(full_teams_df)



        
#%%
import pandas as pd
import calculations as calc

all_pass_combs = [[3,5,2],[3,4,3],[4,3,3], [4,4,2], [4,5,1], [5,3,2], [5,4,1]]
form_name = ["df", "mf", "fw"]
all_combs = []
for comb in all_pass_combs: 
    all_combs = [pd.read_csv("data_cleaned/as/" + form + "/" + str(c) + ".csv") for (c,form) in zip(comb,form_name)]
    all_combs.insert(0,pd.read_csv("data_cleaned/gk.csv"))
    done_df = calc_full_teams(all_combs)
    done_df.to_csv("data_cleaned/as/" + str(comb) + ".csv", index = False)
    
    
#%%
import pandas as pd
import calculations as calc
import numpy as np
import ast
generic = lambda x: ast.literal_eval(x)

conv = {'indexes': generic}

sorted_dfs = cleaners.all_forms_as_df_cleaned()
best_gk = cleaners.clean_gk(sorted_dfs[0])
all_combs =   [best_gk, pd.read_csv("data_cleaned/as/df/4.csv", converters = conv),pd.read_csv("data_cleaned/as/mf/4.csv", converters = conv), pd.read_csv("data_cleaned/as/fw/2.csv", converters = conv)]

all_combs[0]['indexes'] = all_combs[0].index
all_combs[0]['indexes'] = all_combs[0]['indexes'].apply(lambda x: [x])

all_points = calc.calc_from_combs(all_combs, "total_points")
all_costs = calc.calc_from_combs(all_combs, "now_cost" )

points_full = parsers.parse_formations_points_or_cost(all_points)
costs_full = parsers.parse_formations_points_or_cost(all_costs)

sep_ids  = [combs['indexes'].values.tolist() for combs in all_combs]
tot_cost, tot_points, indexes = [],[], []

#%%



def calc_full_teams(all_combs):
    all_combs[0]['indexes'] = all_combs[0].index
    all_combs[0]['indexes'] = all_combs[0]['indexes'].apply(lambda x: [x])

    all_points = calc.calc_from_combs(all_combs, "total_points")
    all_costs = calc.calc_from_combs(all_combs, "now_cost" )

    points_full = parsers.parse_formations_points_or_cost(all_points)
    costs_full = parsers.parse_formations_points_or_cost(all_costs)
    
    it = np.nditer(points_full, flags=['multi_index'])
    bit = np.nditer(costs_full, flags = ['multi_index'])
    sep_ids  = [combs['indexes'].values.tolist() for combs in reversed(all_combs)]
    costs_total, points_total, indexes = [], [], []

    for x, y in zip(it,bit):
        points_total.append(x)
        costs_total.append(y)
        reg_list = [x[it.multi_index[i]] for (i,x) in enumerate(sep_ids)]
        indexes.append([item for sublist in reg_list for item in sublist])
        
    full_teams_df = pd.DataFrame({"cost": costs_total, "points_total": points_total, "indexes": indexes})
    return(full_teams_df)


