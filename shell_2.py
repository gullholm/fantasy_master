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
        
        
#%%
import pandas as pd
import calculations as calc

all_pass_combs = [[3,5,2],[3,4,3],[4,3,3], [4,4,2], [4,5,1], [5,3,2], [5,4,1]]

for comb,name in zip(all_pass_combs, form_name): 
    df = comb[0]
    mf = comb[1]
    fw = comb[2]
    
    all_combs = [pd.read_csv("data_cleaned/as/" + form_name + "/" + c + ".csv") for c in comb]
    
    under_cost =  np.argwhere(costs_full < 500) 
    
#%%
best_gk = cleaners.clean_gk(sorted_dfs[0])
all_combs =   [best_gk, pd.read_csv("data_cleaned/as/df/4.csv"),pd.read_csv("data_cleaned/as/mf/4.csv"), pd.read_csv("data_cleaned/as/fw/2.csv")]

all_combs[0]['indexes'] = all_combs[0].index
all_combs[0]['indexes'] = all_combs[0]['indexes'].apply(lambda x: [x])

all_points = calc.calc_from_combs(all_combs, "total_points")
all_costs = calc.calc_from_combs(all_combs, "now_cost" )

points_full = parsers.parse_formations_points_or_cost(all_points)
costs_full = parsers.parse_formations_points_or_cost(all_costs)

sep_ids  = [combs['indexes'].values.tolist() for combs in all_combs]
tot_cost, tot_points, indexes = [],[], []
