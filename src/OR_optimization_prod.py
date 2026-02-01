import pulp as pl
import pandas as pd
import numpy as np
pd.set_option('future.no_silent_downcasting', True)
from pathlib import Path
import argparse as arg

# Establish filepaths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'

# Create parameter set as a class
class ParameterSet():
    def __init__(self, args: arg.Namespace):
        self.max_or_time = 480
        self.overtime_cost = 120
        self.idle_cost = 60
        self.reschedule_cost = 200
        self.cancellation_cost = 2000
        self.scenario_count = 100
        self.poisson_rates = args.poisson_rates

# Creating MIP as a class
class MIP():
    def __init__(self, params:ParameterSet, surgery_schedule:pd.DataFrame, surgery_durations:list, emergency_scenarios:pd.DataFrame):
        # Initialization
        self.model = pl.LpProblem('MIP', pl.LpMinimize)

        # Data and parameters
        self.a = surgery_schedule
        self.d = surgery_durations
        self.E = emergency_scenarios
        self.ps = params
        
        # Sets
        self.surgeries = list(self.a.columns)
        self.days = list(range(1,len(self.a.index)+2))
        self.scenarios = list(self.E.columns)

    def addVars(self):
        # Surgery scheduling variables
        self.X = pl.LpVariable.dicts('X', ((i, j) for i in self.surgeries for j in self.days), cat='Binary')

        # Overtime minutes
        self.O = pl.LpVariable.dicts('O', ((j, s) for j in self.days for s in self.scenarios), lowBound=0, cat='Continuous')

        # Idle minutes
        self.U = pl.LpVariable.dicts('U', ((j, s) for j in self.days for s in self.scenarios), lowBound=0, cat='Continuous')

        # Surgery re-scheduling variables
        self.R = pl.LpVariable.dicts('R', self.surgeries, lowBound=0, cat='Continuous')

        # Surgery cancellation variables
        self.Y = pl.LpVariable.dicts('Y', self.surgeries, cat='Binary')

    def addObjective(self):
        # Individual costs
        overtime_cost = self.ps.overtime_cost * pl.lpSum((1/self.ps.scenario_count) 
        * self.O[(j,s)] for j in self.days for s in self.scenarios)
        idle_cost = self.ps.idle_cost * pl.lpSum((1/self.ps.scenario_count) * self.U[(j,s)] for j in self.days for s in self.scenarios)
        reschedule_cost = self.ps.reschedule_cost * pl.lpSum(self.R[i] for i in self.surgeries)
        cancellation_cost = self.ps.cancellation_cost * pl.lpSum(self.Y[i] for i in self.surgeries)
        
        # Aggregate objective
        self.model += overtime_cost + idle_cost + reschedule_cost + cancellation_cost

    def addConstraints(self):
        # Overtime minutes
        for j in self.days[:-1]:
            for s in self.scenarios:
                self.model += (self.O[(j,s)] >= pl.lpSum(self.X[(i,j)] * self.d[i] for i in self.surgeries) + 
                              self.E[s][j] - self.ps.max_or_time,
                              f'Overtime_{j}_{s}')

        # Idle minutes
        for j in self.days[:-1]:
            for s in self.scenarios:
                self.model += (self.U[(j,s)] >= self.ps.max_or_time - 
                              pl.lpSum(self.X[(i,j)] * self.d[i] for i in self.surgeries) - self.E[s][j],
                              f'Idle_{j}_{s}')

        # Rescheduled days
        for i in self.surgeries:
            self.model += (self.R[i] >= (pl.lpSum(self.X[(i,j)] * j for j in self.days[:-1]) - 
                          pl.lpSum(self.a[i][j] * j for j in self.days[:-1]) * (1 - self.Y[i])),
                          f'Reschedule_forward_{i}')
            
            self.model += (self.R[i] >= (pl.lpSum(self.a[i][j] * j for j in self.days[:-1]) * 
                          (1 - self.Y[i]) - pl.lpSum(self.X[(i,j)] * j for j in self.days[:-1])),
                          f'Reschedule_backward_{i}')

        # Cancellations
        for i in self.surgeries:
            self.model += (self.Y[i] == self.X[(i, self.days[-1])],
                          f'Cancellation_{i}')

        # Each surgery needs to be assigned a day or cancelled
        for i in self.surgeries:
            self.model += (pl.lpSum(self.X[(i,j)] for j in self.days) == 1,
                          f'Assignment_{i}')

    def solve(self):
        solver = pl.PULP_CBC_CMD(msg=0)
        self.model.solve(solver)

        if self.model.status == pl.LpStatusOptimal:
            return process_output(self)
        elif self.model.status == pl.LpStatusInfeasible:
            self.model.writeLP(SRC_DIR / 'outputs' / 'infeasibility_report.lp')
        else:
            self.model.writeLP(SRC_DIR / 'outputs' / "debugging.lp")

# Generating emergency surgery scenarios
def generate_scenarios(params:ParameterSet, d:list, num_days:int):
    np.random.seed(42)

    # Create empty numpy array to store values
    e = np.zeros((num_days, params.scenario_count))

    # Parametrizing normal distribution for emergency durations
    mu = np.mean(d)
    sigma = np.std(d)

    for j in range(num_days):
        for s in range(params.scenario_count):
            N = np.random.poisson(params.poisson_rates[j])
            if N > 0:
                durations = np.random.normal(
                    loc=mu,
                    scale=sigma,
                    size=N
                )
                durations = np.maximum(durations, 0)
                e[j,s] = durations.sum()
            else:
                e[j,s] = 0
    
    e = pd.DataFrame(e)
    e.index = e.index + 1

    return pd.DataFrame(e)

# Processing MIP output
def process_output(MIP):
    weekday_mapping = {1:'Monday', 2:'Tuesday', 3:'Wednesday', 4:'Thursday', 5:'Friday', 6:''}

    # Creating container dataframes
    solution_binary_df = pd.DataFrame(columns=MIP.surgeries, index=MIP.days)
    simulation_surgery_df = pd.DataFrame(index=MIP.surgeries, columns=['Duration', 'Rescheduled', 'Cancelled'])
    simulation_day_df = pd.DataFrame(index=MIP.days[:-1], columns=['Original', 'Rescheduled'])

    # Creating surgery-level re-scheduling summary
    surgery_output_cols = {
        'Surgery': int,
        'Original Day': str,
        'New Day': str,
        'Rescheduled': str,
        'Cancelled': str
    }
    surgery_output_df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in surgery_output_cols.items()})
    surgery_output_df['Surgery'] = MIP.a.columns
    
    # Getting original surgery days
    for surgery in MIP.a.columns:
        indices = MIP.a.index[MIP.a[surgery] == 1].tolist()
        day = indices[0]
        surgery_output_df.loc[int(surgery), 'Original Day'] = weekday_mapping[int(day)]

    # Getting re-scheduled surgery days
    for (i, j), var in MIP.X.items():
        if pl.value(var) > 0.5:
            surgery_output_df.loc[int(i), 'New Day'] = weekday_mapping[int(j)]
            solution_binary_df.loc[int(j), int(i)] = 1
    
    solution_binary_df = solution_binary_df.fillna(0).infer_objects(copy=False)

    # Indicating surgery rescheduling/cancellations
    surgery_output_df['Cancelled'] = np.where(surgery_output_df['New Day'] == '', 'Yes', 'No')
    surgery_output_df['Rescheduled'] = np.where((surgery_output_df['Original Day'] != surgery_output_df['New Day']) & (surgery_output_df['Cancelled'] != 'Yes'), 'Yes', 'No')

    # Calculating day-level durations and ensuring elective durations match up
    durations_vector = np.array(MIP.d)
    original_durations = MIP.a.to_numpy() @ durations_vector
    new_durations = solution_binary_df.to_numpy() @ durations_vector
    assert(original_durations.sum() == new_durations.sum())

    # Creating simulation surgery input dataframe
    for i, var in MIP.R.items():
        simulation_surgery_df.loc[int(i), 'Rescheduled'] = int(pl.value(var))
    
    for i, var in MIP.Y.items():
        simulation_surgery_df.loc[int(i), 'Cancelled'] = int(pl.value(var))

    simulation_surgery_df['Duration'] = MIP.d

    # Creating simulation day input dataframe
    simulation_day_df['Original'] = original_durations
    simulation_day_df['Rescheduled'] = new_durations[:-1]

    return simulation_surgery_df, simulation_day_df