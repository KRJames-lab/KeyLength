import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from math import pi

# 1. analysis.out 파일 자동 탐색
def find_analysis_out_files(root_dir):
    analysis_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == 'analysis.out':
                analysis_files.append(os.path.join(dirpath, filename))
    return analysis_files

# 2. analysis.out에서 MAE, RMSE, MAPE 추출
def parse_metrics_from_file(filepath):
    metrics = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
    current_skeleton = None
    for line in lines:
        line = line.strip()
        # 스켈레톤 이름: "Nose-Left Eye" 등 (공백 포함, 콜론 없음)
        if line and ':' not in line and 'cm' not in line and 'Data Count' not in line:
            current_skeleton = line
        # MAE, RMSE, MAPE가 있는 줄
        if line.startswith('MAE:') and current_skeleton:
            m = re.match(r'MAE: ([\d.]+)cm, RMSE: ([\d.]+)cm, MAPE: ([\d.]+)%', line)
            if m:
                mae = float(m.group(1))
                rmse = float(m.group(2))
                mape = float(m.group(3))
                metrics[current_skeleton] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    return metrics

# 3. 데이터 정리 및 통합
def collect_all_metrics(root_dir):
    files = find_analysis_out_files(root_dir)
    all_data = {}
    for f in files:
        # 상위 폴더명(예: video_1-upperbody-l) 추출
        key = f.split(os.sep)[1]
        metrics = parse_metrics_from_file(f)
        all_data[key] = metrics
    return all_data

# 4. 시각화 함수
def plot_bar(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # Extract skeleton names
    skeletons = [col[1] for col in df.columns if col[0] == 'MAE']
    for skel in skeletons:
        # For each skeleton, plot MAE, RMSE, MAPE for each directory
        data = df.loc[:, pd.IndexSlice[:, skel]]
        data.columns = data.columns.droplevel(1)  # ('MAE', skel) -> 'MAE'
        fig, ax1 = plt.subplots(figsize=(10,6))
        width = 0.25
        x = np.arange(len(data.index))
        # Bar for MAE, RMSE (left y-axis)
        ax1.bar(x - width/2, data['MAE'], width=width, label='MAE', color='C0')
        ax1.bar(x + width/2, data['RMSE'], width=width, label='RMSE', color='C1')
        ax1.set_ylabel('Value (cm)')
        ax1.set_xlabel('Directory')
        ax1.set_xticks(x)
        ax1.set_xticklabels(data.index)
        ax1.set_title(f'{skel} - MAE, RMSE (cm, left) & MAPE (% , right) by Directory')
        # Bar for MAPE (right y-axis)
        ax2 = ax1.twinx()
        ax2.bar(x + width*1.5, data['MAPE'], width=width, label='MAPE', color='C2', alpha=0.7)
        ax2.set_ylabel('MAPE (%)')
        # Legends
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
        # For each skeleton, get data and normalize (min-max 0~1)
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

# 5. 메인 실행
def main():
    root_dir = '.'
    all_data = collect_all_metrics(root_dir)
    print('all_data:', all_data)
    # Convert to DataFrame
    df = pd.DataFrame.from_dict({(i,j): all_data[i][j] 
                                 for i in all_data.keys() 
                                 for j in all_data[i].keys()}, orient='index')
    df.index = pd.MultiIndex.from_tuples(df.index, names=['Directory', 'Skeleton'])
    df = df.sort_index()
    df = df.unstack('Skeleton')
    print(df.columns)
    # bar plot
    plot_bar(df, './compare/bar')
    # radar plot
    plot_radar(df, './compare/radar')
    print('Done! Results saved in compare/bar and compare/radar')

if __name__ == '__main__':
    main() 