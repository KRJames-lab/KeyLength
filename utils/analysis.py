import numpy as np
import os
import sys
from typing import List, Dict

from .visualization_analysis import save_boxplot, save_histogram, save_framewise_plot

class Analysis:
    """
    Handles statistical analysis and reporting of pose inference results.
    """
    def __init__(self, coco_connections, ground_truth_cm, keypoint_names):
        """
        Initializes the Analysis.

        Args:
            coco_connections (list): List of COCO connection definitions.
            ground_truth_cm (dict): Dictionary of ground truth distances in cm.
            keypoint_names (dict): Dictionary mapping keypoint IDs to names.
        """
        self.COCO_CONNECTIONS = coco_connections
        self.GROUND_TRUTH_CM = ground_truth_cm
        self.keypoint_names = keypoint_names

    def save_results_and_statistics(self, results: List[Dict], results_dir: str, dataset_name: str, output_path: str, trim_ratio: float):
        """Save results and print statistics, and save boxplot/scatterplot visualizations"""
        # Redirect stdout to file
        original_stdout = sys.stdout
        with open(output_path, 'w', encoding='utf-8') as f:
            sys.stdout = f
            
            print(f"\nAnalysis Complete - Statistical Information (centimeter units)")
            print("=" * 60)
            total_frames = len(results)
            valid_frames = len([r for r in results if r['valid_keypoints'] > 0])
            print(f"Overall Statistics:")
            print(f"   - Total Frames: {total_frames}")
            print(f"   - Valid Frames: {valid_frames}")
            print(f"   - Success Rate: {valid_frames/total_frames*100:.1f}%")
            
            self._print_distance_statistics_cm(results, trim_ratio)
            self._print_keypoint_depth_statistics_cm(results, trim_ratio)
        # Restore stdout
        sys.stdout = original_stdout

        # Build errors_dict, gt_dict, pred_dict using COCO_CONNECTIONS
        errors_dict = {c["key"]: [] for c in self.COCO_CONNECTIONS}
        gt_dict = {c["key"]: [] for c in self.COCO_CONNECTIONS}
        pred_dict = {c["key"]: [] for c in self.COCO_CONNECTIONS}
        for result in results:
            for k, v in result['distances_meters'].items():
                if k in self.GROUND_TRUTH_CM and v > 0:
                    gt = self.GROUND_TRUTH_CM[k]
                    pred = v * 100.0
                    
                    # Handle both single value and range GT for error calculation
                    error = 0
                    if isinstance(gt, list):
                        gt_min, gt_max = gt[0], gt[1]
                        if pred < gt_min:
                            error = pred - gt_min  # Negative error
                        elif pred > gt_max:
                            error = pred - gt_max   # Positive error
                    else:
                        error = pred - gt
                    
                    errors_dict[k].append(error)
                    gt_dict[k].append(gt)
                    pred_dict[k].append(pred)
        # Group by group field
        groups = ["head_face", "upper_body", "lower_body"]
        boxplot_dir = os.path.join(results_dir, 'boxplot')
        os.makedirs(boxplot_dir, exist_ok=True)
        histogram_dir = os.path.join(results_dir, 'histogram')
        os.makedirs(histogram_dir, exist_ok=True)
        frameplot_dir = os.path.join(results_dir, 'frameplot')
        os.makedirs(frameplot_dir, exist_ok=True)
        for group_name in groups:
            group_keys = [c["key"] for c in self.COCO_CONNECTIONS if c["group"] == group_name]
            group_errors = {k: errors_dict[k] for k in group_keys if k in errors_dict}
            group_gt = {k: gt_dict[k] for k in group_keys if k in gt_dict}
            group_pred = {k: pred_dict[k] for k in group_keys if k in pred_dict}
            connection_labels = {c["key"]: c["label"] for c in self.COCO_CONNECTIONS if c["group"] == group_name}
            if any(len(v) > 0 for v in group_errors.values()):
                save_boxplot(group_errors, boxplot_dir, title_prefix=f"{group_name}")
                os.rename(
                    os.path.join(boxplot_dir, 'boxplot.png'),
                    os.path.join(boxplot_dir, f'boxplot_{group_name}.png')
                )
                save_histogram(group_errors, histogram_dir, connection_labels, title_prefix=f"{dataset_name} {group_name}")
                save_framewise_plot(group_gt, group_pred, frameplot_dir, connection_labels, title_prefix=f"{dataset_name} {group_name}")

    def _calculate_trimmed_stats(self, data, trim_ratio=0.1):
        """Calculate trimmed statistics (removing top and bottom ratio)"""
        if len(data) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        if len(data) == 1:
            return float(data[0]), 0.0, float(data[0]), float(data[0])
            
        sorted_data = np.sort(data)
        n = len(data)
        trim_size = int(n * trim_ratio)
        
        # Fix: if trim_ratio is 0, use all data
        if trim_size == 0:
            trimmed_data = sorted_data
        elif trim_size * 2 >= n:  # If trimming would remove all data
            trimmed_data = sorted_data
        else:
            trimmed_data = sorted_data[trim_size:-trim_size]
            
        if len(trimmed_data) == 0:  # Additional safety check
            return 0.0, 0.0, 0.0, 0.0
        
        trimmed_mean = np.mean(trimmed_data)
        trimmed_std = np.std(trimmed_data) if len(trimmed_data) > 1 else 0.0
        min_val = np.min(trimmed_data)
        max_val = np.max(trimmed_data)
        
        return trimmed_mean, trimmed_std, min_val, max_val

    def _print_distance_statistics_cm(self, results: List[Dict], trim_ratio: float):
        """Print skeleton connection distance statistics and calculate MAE, RMSE, MAPE against ground truth"""
        # Use COCO_CONNECTIONS order
        mae_dict, rmse_dict, mape_dict = {}, {}, {}
        # Use COCO_CONNECTIONS for order and label
        all_distances = {}
        for result in results:
            for connection_key, distance in result['distances_meters'].items():
                if distance > 0:
                    if connection_key not in all_distances:
                        all_distances[connection_key] = []
                    all_distances[connection_key].append(distance * 100.0)  # Convert to cm
        print(f"\nSkeleton Connection Distance Statistics (centimeter units):")
        print("-" * 60)
        print(f"{trim_ratio * 2 * 100:.0f}% trimmed is applied")
        print()
        
        if 'over_200_counts' not in locals():
            over_200_counts = {}

        for c in self.COCO_CONNECTIONS:
            connection_key = c["key"]
            connection_label = c["label"]
            distances = all_distances.get(connection_key, [])
            distances_array = np.array(distances) if distances else np.array([0.0])
            mean, std, min_val, max_val = self._calculate_trimmed_stats(distances_array, trim_ratio=trim_ratio)
            print(f"{connection_label}")
            print(f"   Average: {mean:.1f}cm")
            print(f"   Standard Deviation: {std:.1f}cm")
            print(f"   Range: {min_val:.1f}cm ~ {max_val:.1f}cm")
            print(f"   Data Count: {len(distances) if distances else 0}")
            gt = self.GROUND_TRUTH_CM.get(connection_key, None)
            
            if gt is not None and len(distances_array) > 0:
                n = len(distances_array)
                trim_size = int(n * trim_ratio)
                if trim_size == 0:
                    trimmed_data = np.sort(distances_array)
                elif trim_size * 2 >= n:
                    trimmed_data = np.sort(distances_array)
                else:
                    trimmed_data = np.sort(distances_array)[trim_size:-trim_size]
                if len(trimmed_data) == 0:
                    trimmed_data = distances_array

                # Check if GT is a range (list) or a single value
                if isinstance(gt, list):
                    # Range-based error calculation
                    gt_min, gt_max = gt[0], gt[1]
                    errors = []
                    percent_errors = []
                    for pred in trimmed_data:
                        error = 0
                        if pred < gt_min:
                            error = gt_min - pred
                            # MAPE calculation based on the closest boundary
                            percent_error = (error / gt_min) * 100 if gt_min != 0 else 0
                        elif pred > gt_max:
                            error = pred - gt_max
                            # MAPE calculation based on the closest boundary
                            percent_error = (error / gt_max) * 100 if gt_max != 0 else 0
                        else: # Inside the range
                            percent_error = 0
                        errors.append(error)
                        percent_errors.append(percent_error)
                    
                    errors = np.array(errors)
                    mae = np.mean(errors)
                    rmse = np.sqrt(np.mean(errors ** 2))
                    mape = np.mean(percent_errors)
                    ape = np.array(percent_errors) # Use the already calculated percentage errors
                    over_200_count = int(np.sum(ape > 200))

                else:
                    # Single value error calculation (existing logic)
                abs_err = np.abs(trimmed_data - gt)
                mae = np.mean(abs_err)
                rmse = np.sqrt(np.mean((trimmed_data - gt) ** 2))
                mape = np.mean(abs_err / gt * 100) if gt != 0 else 0.0
                # Count the number of frames with APE over 200%
                ape = abs_err / gt * 100 if gt != 0 else np.zeros_like(abs_err)
                over_200_count = int(np.sum(ape > 200))
            else:
                mae = rmse = mape = 0.0
                over_200_count = 0

            print(f"   MAE: {mae:.2f}cm, MAPE: {mape:.2f}%({over_200_count}), RMSE: {rmse:.2f}cm")
            mae_dict[connection_key] = mae
            rmse_dict[connection_key] = rmse
            mape_dict[connection_key] = mape
            over_200_counts[connection_key] = over_200_count
            print()

        # Print overall average error using trimmed means
        all_mae = np.mean(list(mae_dict.values()))
        all_rmse = np.mean(list(rmse_dict.values()))
        all_mape = np.mean(list(mape_dict.values()))
        total_over_200 = sum(over_200_counts.values())
        print("Overall Connection Average Error:")
        print(f"   MAE: {all_mae:.2f}cm, MAPE: {all_mape:.2f}%({total_over_200}), RMSE: {all_rmse:.2f}cm")

    def _print_keypoint_depth_statistics_cm(self, results: List[Dict], trim_ratio: float):
        """Print keypoint depth statistics (in centimeter units)"""
        keypoint_depths = {i: [] for i in self.keypoint_names.keys()}
        for result in results:
            keypoints_3d = np.array(result['keypoints_3d'])
            for i, (px, py, depth, conf) in enumerate(keypoints_3d):
                if conf > 0.3 and depth > 0:
                    keypoint_depths[i].append(depth * 100.0)  # Convert to cm
        
        print(f"\nKeypoint Depth Statistics (centimeter units):")
        print("-" * 60)
        print(f"{trim_ratio * 2 * 100:.0f}% trimmed is applied")
        print()
        
        for joint_id, depths in keypoint_depths.items():
            if depths:
                depths_array = np.array(depths)
                joint_name = self.keypoint_names.get(joint_id, f"Joint{joint_id}")
                
                # Calculate trimmed statistics
                mean, std, min_val, max_val = self._calculate_trimmed_stats(depths_array, trim_ratio=trim_ratio)
                
                print(f"{joint_name} (ID: {joint_id})")
                print(f"   Average Depth: {mean:.1f}cm")
                print(f"   Standard Deviation: {std:.1f}cm")
                print(f"   Range: {min_val:.1f}cm ~ {max_val:.1f}cm")
                print(f"   Detection Count: {len(depths_array)}")
                print() 