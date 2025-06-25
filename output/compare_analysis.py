import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from math import pi
import argparse

# Find analysis.out
def find_analysis_out_files(root_dir):
    analysis_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == 'analysis.out':
                analysis_files.append(os.path.join(dirpath, filename))
    return analysis_files

# Extract MAE, RMSE, MAPE from analysis.out
def parse_metrics_from_file(filepath):
    metrics = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
    current_skeleton = None
    for line in lines:
        line = line.strip()

        if line and ':' not in line and not any(keyword in line for keyword in ['cm', 'Data Count', 'Average', 'Standard', 'Range', 'Overall', 'applied', 'Statistics', '---']):
            current_skeleton = line.strip()

        if line.startswith('MAE:') and current_skeleton:
            mae_match = re.search(r'MAE: ([\d.]+)cm', line)
            rmse_match = re.search(r'RMSE: ([\d.]+)cm', line)
            mape_match = re.search(r'MAPE: ([\d.]+)%', line)

            if mae_match and rmse_match and mape_match:
                mae = float(mae_match.group(1))
                rmse = float(rmse_match.group(1))
                mape = float(mape_match.group(1))
                metrics[current_skeleton] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
                current_skeleton = None
    return metrics

# Proces extracted data
def collect_all_metrics(root_dir, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = []
    files = find_analysis_out_files(root_dir)
    all_data = {}
    for f in files:
        relative_path = os.path.relpath(os.path.dirname(f), root_dir)
        key = relative_path.split(os.sep)[0]

        if key in exclude_dirs:
            print(f"Skipping excluded directory: {key}")
            continue
        
        metrics = parse_metrics_from_file(f)

        if key not in all_data:
            all_data[key] = {}
        
        all_data[key].update(metrics)
    return all_data

# Visualization
def plot_bar(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    skeletons = [col[1] for col in df.columns if col[0] == 'MAE']
    for skel in skeletons:
        data = df.loc[:, pd.IndexSlice[:, skel]]
        data.columns = data.columns.droplevel(1)
        fig, ax1 = plt.subplots(figsize=(10,6))
        width = 0.25
        x = np.arange(len(data.index))
        ax1.bar(x - width/2, data['MAE'], width=width, label='MAE', color='C0')
        ax1.bar(x + width/2, data['RMSE'], width=width, label='RMSE', color='C1')
        ax1.set_ylabel('Value (cm)')
        ax1.set_xlabel('Directory')
        ax1.set_xticks(x)
        ax1.set_xticklabels(data.index)
        ax1.set_title(f'{skel} - MAE, RMSE (cm, left) & MAPE (% , right) by Directory')
        ax2 = ax1.twinx()
        ax2.bar(x + width*1.5, data['MAPE'], width=width, label='MAPE', color='C2', alpha=0.7)
        ax2.set_ylabel('MAPE (%)')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{skel}_bar.png'))
        plt.close()

def plot_radar(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    metrics = ['MAE', 'RMSE', 'MAPE']
    skeletons = [col[1] for col in df.columns if col[0] == 'MAE']
    for skel in skeletons:
        data = df.loc[:, pd.IndexSlice[:, skel]]
        data.columns = data.columns.droplevel(1)
        normed = (data - data.min()) / (data.max() - data.min())
        dirs = normed.index
        num_vars = len(metrics)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]
        plt.figure(figsize=(6,6))
        for idx in dirs:
            values = [normed.loc[idx][m] for m in metrics]
            values += values[:1]
            plt.polar(angles, values, label=idx, marker='o')
        plt.xticks(angles[:-1], metrics)
        plt.title(f'{skel} - Normalized MAE, RMSE, MAPE Radar Chart')
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{skel}_radar.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize metrics from 'analysis.out' files.")
    parser.add_argument('--exclude', nargs='+', help='List of directories to exclude from analysis.')
    args = parser.parse_args()

    root_dir = '.'
    all_data = collect_all_metrics(root_dir, exclude_dirs=args.exclude)
    
    if not all_data:
        print("No data could be collected. Exiting.")
        return

    df = pd.DataFrame.from_dict({(i,j): all_data[i][j] 
                                 for i in all_data.keys() 
                                 for j in all_data[i].keys()}, orient='index')

    if df.empty:
        print("DataFrame is empty. Cannot generate plots.")
        return

    df.index = pd.MultiIndex.from_tuples(df.index, names=['Directory', 'Skeleton'])
    df = df.sort_index()
    df = df.unstack('Skeleton')
    
    if df.empty:
        print("Unstacked DataFrame is empty. Cannot generate plots.")
        return

    # bar plot
    plot_bar(df, './compare/bar')
    # radar plot
    plot_radar(df, './compare/radar')
    print('Done! Results saved in compare/bar and compare/radar')

if __name__ == '__main__':
    main() 