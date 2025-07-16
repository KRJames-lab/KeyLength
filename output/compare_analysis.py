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

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
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
                    
                    metrics['Total Connection Metrics'] = overall_metrics

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
    # 에러 지표와 PR 지표를 각각 저장할 하위 디렉터리 생성
    error_dir = os.path.join(save_dir, 'error')
    pr_dir = os.path.join(save_dir, 'pr')
    os.makedirs(error_dir, exist_ok=True)
    os.makedirs(pr_dir, exist_ok=True)
    skeletons = [col[1] for col in df.columns if col[0] == 'MAE']
    
    has_pr = 'Precision' in [col[0] for col in df.columns]

    for skel in skeletons:
        data = df.loc[:, pd.IndexSlice[:, skel]]
        data.columns = data.columns.droplevel(1)

        x = np.arange(len(data.index))

        # --------------------
        # Error metrics plot (MAE, RMSE, MAPE)
        # --------------------
        fig_err, ax_err1 = plt.subplots(figsize=(12, 7))
        ax_err1.plot(x, data['MAE'], marker='o', linestyle='-', label='MAE', color='C0')
        ax_err1.plot(x, data['RMSE'], marker='s', linestyle='--', label='RMSE', color='C1')
        ax_err1.set_ylabel('MAE / RMSE (cm)')
        ax_err1.set_xlabel('Models')
        ax_err1.set_xticks(x)
        ax_err1.set_xticklabels(data.index, rotation=0, ha="center")
        ax_err1.set_title(f'{skel} - Error Metrics')
        # 주요/보조 격자 모두 표시하여 더 촘촘한 그리드
        ax_err1.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.8)
        ax_err1.minorticks_on()
        ax_err1.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)

        ax_err2 = ax_err1.twinx()
        ax_err2.plot(x, data['MAPE'], marker='^', linestyle='-.', label='MAPE', color='C2')
        ax_err2.set_ylabel('MAPE (%)')
        # 보조 축에도 격자 추가 (MAPE)
        ax_err2.grid(False)  # 보조 축 격자는 주축과 겹치므로 끔

        # 축 반전(오차가 작을수록 위로 보이도록)
        ax_err1.invert_yaxis()
        ax_err2.invert_yaxis()

        h1e, l1e = ax_err1.get_legend_handles_labels()
        h2e, l2e = ax_err2.get_legend_handles_labels()
        ax_err1.legend(h1e + h2e, l1e + l2e, loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(error_dir, f'{skel}_error.png'))
        plt.close()

        # --------------------
        # Precision / Recall plot (if available)
        # --------------------
        if has_pr and {'Precision', 'Recall'}.issubset(set(data.columns)):
            fig_pr, ax_pr = plt.subplots(figsize=(12, 7))

            # Precision / Recall 전용 플롯 (다른 지표는 포함하지 않도록 명시적으로 컬럼 선택)
            pr_data = data[['Precision', 'Recall']]
            ax_pr.plot(x, pr_data['Precision'], marker='d', linestyle='--', label='Precision', color='C3')
            ax_pr.plot(x, pr_data['Recall'],   marker='x', linestyle='-',  label='Recall',    color='C4')
            ax_pr.set_ylabel('Value (%)')
            ax_pr.set_xlabel('Models')
            ax_pr.set_xticks(x)
            ax_pr.set_xticklabels(data.index, rotation=0, ha="center")
            ax_pr.set_title(f'{skel} - Precision / Recall')
            # 주요/보조 격자 모두 표시하여 더 촘촘하게
            ax_pr.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.8)
            ax_pr.minorticks_on()
            ax_pr.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)
            ax_pr.legend(loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(pr_dir, f'{skel}_pr.png'))
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
    parser.add_argument('--order', nargs='+', help='Custom order of directories to display in plots.')
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

    # 사용자 지정 순서 적용
    if args.order:
        specified = [d for d in args.order if d in df.index]
        if len(specified) < len(args.order):
            missing = set(args.order) - set(specified)
            if missing:
                print(f"Warning: the following directories specified in --order were not found and will be ignored: {', '.join(missing)}")
        remaining = [d for d in df.index if d not in specified]
        df = df.reindex(specified + remaining)
    
    if df.empty:
        print("Unstacked DataFrame is empty. Cannot generate plots.")
        return

    # line plot
    plot_bar(df, './compare/line')
    # radar plot
    plot_radar(df, './compare/radar')
    print('Done! Results saved in compare/line and compare/radar')

if __name__ == '__main__':
    main() 