
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, value

def solve_optimization_problem(data, budget_constraint):

    # Create a linear programming problem
    prob = LpProblem("VectorSelection", LpMaximize)

    # Decision variables
    vectors = list(range(len(data)))
    x = LpVariable.dicts("x", vectors, cat="Binary")

    # Objective function
    prob += lpSum(data[i][5] * x[i] for i in vectors), "Objective"

    # Formation constraints
    prob += lpSum(data[i][0] * x[i] for i in vectors) == 1, "GoalieConstraint"
    prob += lpSum(data[i][1] * x[i] for i in vectors) >= 3, "DefenderConstraint"
    prob += lpSum(data[i][1] * x[i] for i in vectors) <= 5, "DefenderConstraint2"
    prob += lpSum(data[i][2] * x[i] for i in vectors) >= 3, "MidfielderConstraint"
    prob += lpSum(data[i][2] * x[i] for i in vectors) <= 5, "MidfielderConstraint2"
    prob += lpSum(data[i][3] * x[i] for i in vectors) >= 1, "FowardConstraint"
    prob += lpSum(data[i][3] * x[i] for i in vectors) <= 3, "ForwardConstraint2"

    # Budget constraint
    prob += lpSum(data[i][4] * x[i] for i in vectors) <= budget_constraint, "BudgetConstraint"

    # Constraint to select exactly 11 vectors
    prob += lpSum(x[i] for i in vectors) == 11, "TotalPlayersConstraint"
    prob.solve()

    if prob.status == 1: # If the optimization problem is feasible
        return int(value(prob.objective))
    else:
        return None

def get_players(filepath):
    data = pd.read_csv(filepath, usecols=['element_type', 'now_cost', 'total_points', 'id'])
    players = [
        [1 if i == row['element_type'] - 1 else 0 for i in range(4)] + [row['now_cost'], row['total_points'], row['id']] for _, row in data.iterrows()
    ]
    return players

def calculate_optimal_team_scores(players):
    obj_values = []  
    for budget_constraint in range(500, 1001, 50):
        obj_value = solve_optimization_problem(players, budget_constraint)
        obj_values.append(obj_value)
    return obj_values        

if __name__ == "__main__":
    players = get_players('data/pl_csv/players_raw_2021.csv')
    obj_values = calculate_optimal_team_scores(players)
    print(obj_values)