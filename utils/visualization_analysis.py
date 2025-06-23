import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def save_boxplot(errors_dict, output_dir, title_prefix=""):
    """
    Save boxplot of error distribution for each connection.
    errors_dict: {connection(str): [error1, error2, ...]}
    output_dir: directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    connections = list(errors_dict.keys())
    data = [errors_dict[c] for c in connections]
    plt.figure(figsize=(max(10, len(connections)*0.7), 6))
    sns.boxplot(data=data)
    plt.xticks(range(len(connections)), connections, rotation=45)
    plt.ylabel('Error (cm)')
    plt.title(f'{title_prefix} : Error Distribution (Boxplot)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot.png'))
    plt.close()

def save_histogram(errors_dict, output_dir, connection_labels, title_prefix=""):
    """
    Save histogram of error values for each connection.
    errors_dict: {connection(str): [error1, error2, ...]}
    output_dir: directory to save the plots
    connection_labels: {connection(str): label(str)}
    """
    os.makedirs(output_dir, exist_ok=True)
    for conn, errors in errors_dict.items():
        if len(errors) == 0:
            continue
        label = connection_labels.get(conn, conn)
        plt.figure(figsize=(6, 4))
        plt.hist(errors, bins=20, alpha=0.75, color='blue', edgecolor='black')
        plt.xlabel('Error (cm)')
        plt.ylabel('Frequency')
        plt.title(f'{conn} {label} : Error Histogram')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'histogram_{conn}.png'))
        plt.close()

def save_framewise_plot(gt_dict, pred_dict, output_dir, connection_labels, title_prefix=""):
    """
    Save framewise scatter plot of measured values and ground truth for each connection.
    gt_dict: {connection(str): [gt, gt, ...]}
    pred_dict: {connection(str): [pred, pred, ...]}
    output_dir: directory to save the plots
    connection_labels: {connection(str): label(str)}
    """
    os.makedirs(output_dir, exist_ok=True)
    for conn in pred_dict.keys():
        pred = np.array(pred_dict[conn])
        gt_data = gt_dict.get(conn, [])
        if len(pred) == 0 or len(gt_data) == 0:
            continue
            
        label = connection_labels.get(conn, conn)
        frame_indices = np.arange(len(pred))
        plt.figure(figsize=(10, 4))
        plt.scatter(frame_indices, pred, label='Measured', color='blue', s=10) # s for marker size
        
        # Check if GT is a range or a single line
        if gt_data and isinstance(gt_data[0], list):
            # GT is a range [min, max]
            gt_min = np.array([item[0] for item in gt_data])
            gt_max = np.array([item[1] for item in gt_data])
            plt.fill_between(frame_indices, gt_min, gt_max, color='red', alpha=0.3, label='Ground Truth Range')
        else:
            # GT is a single line
            gt = np.array(gt_data)
        plt.plot(frame_indices, gt, label='Ground Truth', linestyle='--', color='red')

        plt.xlabel('Frame')
        plt.ylabel('Length (cm)')
        plt.title(f'{conn} {label} : Measured vs Ground Truth')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'frameplot_{conn}.png'))
        plt.close() 