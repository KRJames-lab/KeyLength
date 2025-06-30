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
    for i, line in enumerate(lines):
        line = line.strip()

        if line and ':' not in line and not any(keyword in line for keyword in ['cm', 'Data Count', 'Average', 'Standard', 'Range', 'Overall', 'applied', 'Statistics', '---']):
            current_skeleton = line.strip()

        if line.startswith('MAE:') and current_skeleton:
            mae_match = re.search(r'MAE: ([\d.]+)cm', line)
            rmse_match = re.search(r'RMSE: ([\d.]+)cm', line)
            mape_match = re.search(r'MAPE: ([\d.]+)%', line)

            if mae_match and rmse_match and mape_match:
                metrics[current_skeleton] = {
                    'MAE': float(mae_match.group(1)),
                    'RMSE': float(rmse_match.group(1)),
                    'MAPE': float(mape_match.group(1))
                }
                if i + 1 < len(lines) and lines[i+1].strip().startswith('Precision:'):
                    next_line = lines[i+1].strip()
                    precision_match = re.search(r'Precision: ([\d.]+)%', next_line)
                    recall_match = re.search(r'Recall: ([\d.]+)%', next_line)
                    if precision_match and recall_match:
                        metrics[current_skeleton]['Precision'] = float(precision_match.group(1))
                        metrics[current_skeleton]['Recall'] = float(recall_match.group(1))
                current_skeleton = None
        
        if line.startswith('Overall Connection Average Error:'):
            if i + 2 < len(lines):
                mae_line = lines[i+1].strip()
                pr_line = lines[i+2].strip()

                mae_match = re.search(r'MAE: ([\d.]+)cm', mae_line)
                rmse_match = re.search(r'RMSE: ([\d.]+)cm', mae_line)
                mape_match = re.search(r'MAPE: ([\d.]+)%', mae_line)
                
                precision_match = re.search(r'Overall Precision: ([\d.]+)%', pr_line)
                recall_match = re.search(r'Recall: ([\d.]+)%', pr_line)

                if mae_match and rmse_match and mape_match:
                    overall_metrics = {
                        'MAE': float(mae_match.group(1)),
                        'RMSE': float(rmse_match.group(1)),
                        'MAPE': float(mape_match.group(1))
                    }
                    if precision_match and recall_match:
                        overall_metrics['Precision'] = float(precision_match.group(1))
                        overall_metrics['Recall'] = float(recall_match.group(1))
                    
                    metrics['Overall'] = overall_metrics

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
    
    has_pr = 'Precision' in [col[0] for col in df.columns]

    for skel in skeletons:
        data = df.loc[:, pd.IndexSlice[:, skel]]
        data.columns = data.columns.droplevel(1)
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        num_metrics_left = 2 # MAE, RMSE
        num_metrics_right = 1 + (2 if has_pr else 0) # MAPE, Precision, Recall
        total_bars = num_metrics_left + num_metrics_right
        width = 0.8 / total_bars

        x = np.arange(len(data.index))
        
        # Left Y-axis: MAE, RMSE
        ax1.bar(x - width * 1.5, data['MAE'], width=width, label='MAE', color='C0')
        ax1.bar(x - width * 0.5, data['RMSE'], width=width, label='RMSE', color='C1')
        ax1.set_ylabel('Value (cm)')
        ax1.set_xlabel('Directory')
        ax1.set_xticks(x)
        ax1.set_xticklabels(data.index, rotation=45, ha="right")
        ax1.set_title(f'{skel}')
        
        # Right Y-axis: MAPE, Precision, Recall
        ax2 = ax1.twinx()
        ax2.bar(x + width * 0.5, data['MAPE'], width=width, label='MAPE', color='C2')
        if has_pr:
            ax2.bar(x + width * 1.5, data['Precision'], width=width, label='Precision', color='C3')
            ax2.bar(x + width * 2.5, data['Recall'], width=width, label='Recall', color='C4')
        ax2.set_ylabel('Value (%)')
        
        # Legends
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{skel}_bar.png'))
        plt.close()

def plot_radar(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    radar_metrics = ['MAPE', 'Precision', 'Recall']

    skeletons = [col[1] for col in df.columns if col[0] == 'MAE']
    
    for skel in skeletons:
        data = df.loc[:, pd.IndexSlice[:, skel]]
        data.columns = data.columns.droplevel(1)
        
        metrics_to_plot = [m for m in radar_metrics if m in data.columns]
        
        if not metrics_to_plot:
            continue

        normed = (data[metrics_to_plot] - data[metrics_to_plot].min()) / (data[metrics_to_plot].max() - data[metrics_to_plot].min())
        normed.fillna(0, inplace=True)
        
        dirs = normed.index
        num_vars = len(metrics_to_plot)
        
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]
        
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        
        for idx in dirs:
            values = normed.loc[idx].tolist()
            values += values[:1]
            ax.plot(angles, values, label=idx, marker='o')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_to_plot)
        plt.title(f'{skel} - Normalized Metrics Radar Chart', size=15, y=1.1)
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