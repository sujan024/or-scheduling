import pandas as pd
import numpy as np
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
        self.simulation_count = 10
        self.poisson_rates = args.poisson_rates

# Calculate surgery change costs
def calculate_surgery_costs(params:ParameterSet, surgery_df:pd.DataFrame) -> float:
    total_reschedule_cost = params.reschedule_cost * surgery_df['Rescheduled'].sum()
    total_cancellation_cost = params.cancellation_cost * surgery_df['Cancelled'].sum()
    
    return total_reschedule_cost + total_cancellation_cost

def run_simulation(params:ParameterSet, surgery_df:pd.DataFrame, day_df:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_change_cost = calculate_surgery_costs(params, surgery_df)
    
    # Creating container dataframes
    original_results_df = pd.DataFrame(index=day_df.index, columns=['Overtime Cost','Idle Cost'])
    rescheduled_results_df = pd.DataFrame(index=day_df.index, columns=['Overtime Cost','Idle Cost'])

    original_overtime_df = pd.DataFrame(index=day_df.index, columns=range(params.simulation_count))
    original_idle_df = pd.DataFrame(index=day_df.index, columns=range(params.simulation_count))
    rescheduled_overtime_df = pd.DataFrame(index=day_df.index, columns=range(params.simulation_count))
    rescheduled_idle_df = pd.DataFrame(index=day_df.index, columns=range(params.simulation_count))

    np.random.seed(42)

    num_days = day_df.shape[0]

    d = surgery_df['Duration']
    mu = np.mean(d)
    sigma = np.std(d)

    # Simulation
    for s in range(params.simulation_count):
        for j in range(num_days):
            N = np.random.poisson(params.poisson_rates[j])
            if N > 0:
                durations = np.random.normal(
                    loc=mu,
                    scale=sigma,
                    size=N
                )
                durations = np.maximum(durations, 0)
                total_emergency_duration = durations.sum()
            else:
                total_emergency_duration = 0

            original_total_surgery_duration = day_df.loc[j+1,'Original'] + total_emergency_duration
            original_overtime_df.loc[j+1,s] = params.overtime_cost * max(original_total_surgery_duration - params.max_or_time, 0)
            original_idle_df.loc[j+1,s] = params.idle_cost * max(params.max_or_time - original_total_surgery_duration, 0)

            rescheduled_total_surgery_duration = day_df.loc[j+1,'Rescheduled'] + total_emergency_duration
            rescheduled_overtime_df.loc[j+1,s] = params.overtime_cost * max(rescheduled_total_surgery_duration - params.max_or_time, 0)
            rescheduled_idle_df.loc[j+1,s] = params.idle_cost * max(params.max_or_time - rescheduled_total_surgery_duration, 0)

    original_results_total = (original_overtime_df + original_idle_df).sum()
    rescheduled_results_total = (rescheduled_overtime_df + rescheduled_idle_df).sum() + total_change_cost
    total_cost_comparison_df = pd.concat([original_results_total, rescheduled_results_total], axis=1)
    total_cost_comparison_df.columns = ['Original','New']

    original_results_df['Overtime Cost'] = original_overtime_df.mean(axis=1)
    original_results_df['Idle Cost'] = original_idle_df.mean(axis=1)
    rescheduled_results_df['Overtime Cost'] = rescheduled_overtime_df.mean(axis=1)
    rescheduled_results_df['Idle Cost'] = rescheduled_idle_df.mean(axis=1)

    return original_results_df, rescheduled_results_df, total_cost_comparison_df