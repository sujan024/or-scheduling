from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from pathlib import Path
import OR_optimization_prod
import OR_simulation_prod
import pulp as pl
import webbrowser
import threading
import time

# Initiate Flask app
app = Flask(__name__)

# Establish filepaths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'

# Define parameters for modeling and simulation
class ParameterSet():
    def __init__(self, data):
        self.max_or_time = 480
        self.overtime_cost = data.get('overtime_cost', 120)
        self.idle_cost = data.get('idle_cost', 60)
        self.reschedule_cost = data.get('reschedule_cost', 200)
        self.cancellation_cost = data.get('cancellation_cost', 2000)
        self.scenario_count = data.get('scenario_count', 100)
        self.simulation_count = data.get('simulation_count', 100)
        self.poisson_rates = data.get('poisson_rates', [1, 1, 1, 1, 1])

# Opening webpage
@app.route('/')
def serve_index():
    """Serve the HTML file"""
    return send_from_directory(SRC_DIR, 'index.html')

# Accessing CSS files
@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS, etc.)"""
    return send_from_directory(SRC_DIR, filename)

# Getting the available week-suite combinations
@app.route('/api/weeks-suites', methods=['GET'])
def get_weeks_suites():
    """Get available weeks and suites from the data"""
    try:
        data = pd.read_csv(DATA_DIR / 'cleaned_data.csv')
        week_suites = sorted(data['week_suite'].unique().tolist())
        
        # Parse into weeks and suites
        weeks = list(set([ws.split('_')[0] for ws in week_suites]))
        suites = list(set([ws.split('_')[1] for ws in week_suites]))
        
        # Sort weeks and suites numerically
        weeks_sorted = sorted(weeks, key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
        suites_sorted = sorted(suites, key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
        
        return jsonify({
            'weeks': weeks_sorted,
            'suites': suites_sorted,
            'week_suites': week_suites
        })
    
    except Exception as e:
        print(e)
        return jsonify({'error': "Internal Server Error"}), 500

# Displaying initial schedule data without optimization
@app.route('/api/initial-data', methods=['POST'])
def get_initial_data():
    """Get initial schedule data for selected week and suite"""
    try:
        data = request.json
        week = data.get('week')
        or_suite = data.get('or_suite')
        
        if not week or not or_suite:
            return jsonify({'error': 'Week and suite are required'}), 400
        
        # Load and filter data
        df = pd.read_csv(DATA_DIR / 'cleaned_data.csv')
        filtered_data = df[df['week_suite'] == f"{week}_{or_suite}"].reset_index()
        filtered_data.drop(columns=filtered_data.columns[[0,1]], inplace=True)
        specialty = " & ".join(filtered_data['service'].unique())
        
        if filtered_data.empty:
            return jsonify({'error': 'No surgeries found for selected week and suite'}), 400
        
        weekday_mapping = {1:'Monday', 2:'Tuesday', 3:'Wednesday', 4:'Thursday', 5:'Friday'}
        
        # Get surgery schedule and details
        surgery_list = []
        for idx, row in filtered_data.iterrows():
            surgery_list.append({
                'surgery_id': int(idx),
                'weekday': weekday_mapping.get(row['weekday'], ''),
                'weekday_num': int(row['weekday']),
                'code': row['cpt_code'],
                'procedure': row['cpt_description'],
                'duration': float(row['booked_time'])
            })
        
        # Calculate day-level summaries
        day_summary = filtered_data.groupby('weekday')['booked_time'].sum().to_dict()
        day_data = []
        for day_num in sorted(day_summary.keys()):
            day_data.append({
                'day': weekday_mapping.get(day_num, ''),
                'total_duration': float(day_summary[day_num])
            })
        
        return jsonify({
            'success': True,
            'surgeries': surgery_list,
            'day_summary': day_data,
            'specialty': specialty
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        return jsonify({'error': "Internal server error"}), 500

# Run optimization and simulation
@app.route('/api/optimize', methods=['POST'])
def optimize_schedule():
    """Run optimization and simulation"""
    try:
        data = request.json
        
        # Create parameter set
        params = ParameterSet(data)
        
        # Get surgery schedule
        week = data.get('week')
        or_suite = data.get('or_suite')
        
        # Load and filter data
        df = pd.read_csv(DATA_DIR / 'cleaned_data.csv')
        filtered_data = df[df['week_suite'] == f"{week}_{or_suite}"].reset_index()
        filtered_data.drop(columns=filtered_data.columns[[0,1]], inplace=True)
        specialty = " & ".join(filtered_data['service'].unique())
        
        if filtered_data.empty:
            return jsonify({'error': 'No surgeries found for selected week and suite'}), 400
        
        # Prepare optimization inputs
        d_surgery_durations = list(filtered_data['booked_time'])
        dummies = pd.get_dummies(filtered_data['weekday'], dtype=int)
        a_surgery_schedule = dummies.T
        
        # Generate emergency scenarios
        e_emergency_scenarios = OR_optimization_prod.generate_scenarios(
            params, 
            d_surgery_durations, 
            a_surgery_schedule.shape[0]
        )
        
        # Run optimization
        mip = OR_optimization_prod.MIP(params, a_surgery_schedule, d_surgery_durations, e_emergency_scenarios)
        mip.addVars()
        mip.addObjective()
        mip.addConstraints()
        simulation_surgery_df, simulation_day_df = mip.solve()
        
        # Calculate base rescheduling cost
        surgery_rescheduling_cost = OR_simulation_prod.calculate_surgery_costs(params, simulation_surgery_df)
        
        # Run simulation
        original_results, rescheduled_results, total_cost_results = OR_simulation_prod.run_simulation(
            params, 
            simulation_surgery_df, 
            simulation_day_df
        )

        # Get percentage of simulations where rescheduled outperformed original
        total_cost_results['Improvement'] = total_cost_results['New'] < total_cost_results['Original']
        cost_improvement_count = total_cost_results['Improvement'].sum()
        
        # Prepare response data
        weekday_mapping = {1:'Monday', 2:'Tuesday', 3:'Wednesday', 4:'Thursday', 5:'Friday'}
        
        # Surgery schedule modifications
        surgery_changes = []
        for idx in simulation_surgery_df.index:
            # Get original day
            original_day_idx = a_surgery_schedule.index[a_surgery_schedule[idx] == 1].tolist()[0]
            original_day = weekday_mapping.get(original_day_idx, '')
            
            # Get new day from optimization
            new_day = original_day
            if simulation_surgery_df.loc[idx, 'Cancelled'] == 1:
                new_day = 'Cancelled'
            elif simulation_surgery_df.loc[idx, 'Rescheduled'] > 0:
                # Find which day it was rescheduled to
                for d in range(1, 6):
                    if pl.value(mip.X[(idx, d)]) > 0.5:
                        new_day = weekday_mapping[d]
                        break
            
            surgery_changes.append({
                'surgery_id': int(idx),
                'duration': float(simulation_surgery_df.loc[idx, 'Duration']),
                'original_day': original_day,
                'new_day': new_day,
                'rescheduled': 'Yes' if simulation_surgery_df.loc[idx, 'Rescheduled'] > 0 else 'No',
                'cancelled': 'Yes' if simulation_surgery_df.loc[idx, 'Cancelled'] == 1 else 'No'
            })
        
        # Day-level summary
        day_summary = []
        for day_idx in simulation_day_df.index:
            day_summary.append({
                'day': weekday_mapping.get(day_idx, ''),
                'original_duration': float(simulation_day_df.loc[day_idx, 'Original']),
                'new_duration': float(simulation_day_df.loc[day_idx, 'Rescheduled']),
                'expected_emergency': float(e_emergency_scenarios.mean()[day_idx])
            })
        
        # Simulation results
        simulation_results = []
        for day_idx in original_results.index:
            simulation_results.append({
                'day': weekday_mapping.get(day_idx, ''),
                'original_overtime': float(original_results.loc[day_idx, 'Overtime Cost']),
                'original_idle': float(original_results.loc[day_idx, 'Idle Cost']),
                'rescheduled_overtime': float(rescheduled_results.loc[day_idx, 'Overtime Cost']),
                'rescheduled_idle': float(rescheduled_results.loc[day_idx, 'Idle Cost'])
            })
        
        # Calculate totals
        total_original_cost = original_results.sum().sum()
        total_rescheduled_cost = rescheduled_results.sum().sum() + surgery_rescheduling_cost
        cost_improvement = total_original_cost - total_rescheduled_cost
        
        return jsonify({
            'success': True,
            'surgery_changes': surgery_changes,
            'specialty': specialty,
            'day_summary': day_summary,
            'simulation_results': simulation_results,
            'simulation_count': params.simulation_count,
            'totals': {
                'rescheduling_cost': float(surgery_rescheduling_cost),
                'total_original_cost': float(total_original_cost),
                'total_rescheduled_cost': float(total_rescheduled_cost),
                'cost_improvement': float(cost_improvement),
                'cost_improvement_count': float(cost_improvement_count)
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        return jsonify({'error': "Internal Server Error"}), 500

def open_browser():
    """Open browser after server starts"""
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))

    if not os.environ.get("RENDER"):
        print("Running locally...")
        threading.Thread(target=open_browser, daemon=True).start()

    app.run(host='0.0.0.0', port=port, debug=False)