import cv2
import numpy as np
import onnxruntime as ort
import argparse
import os
from typing import Tuple, List, Dict
import json
import sys

from utils import PoseVisualizer, KeypointFilter, ImagePreprocessor, DepthProcessor

class ImageDepthPoseInference:
    def __init__(self, onnx_model_path: str):
        """Initialize the 3D pose inference system.

        Args:
            onnx_model_path (str): ONNX model file path
        """
        self.onnx_model_path = onnx_model_path
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = None
        
        # COCO format 17 keypoints
        self.original_keypoints = list(range(17))
        
        # Standard COCO format 17 keypoint names
        self.keypoint_names = {
            0: "Nose", 1: "Left Eye", 2: "Right Eye", 3: "Left Ear", 4: "Right Ear",
            5: "Left Shoulder", 6: "Right Shoulder", 7: "Left Elbow", 8: "Right Elbow", 
            9: "Left Wrist", 10: "Right Wrist", 11: "Left Pelvis", 12: "Right Pelvis",
            13: "Left Knee", 14: "Right Knee", 15: "Left Ankle", 16: "Right Ankle"
        }
        
        # Skeleton connections
        self.skeleton_connections = [
            (0, 1), (0, 2),     # Nose-Eyes
            (1, 3), (2, 4),     # Eyes-Ears
            (1, 2),             # Eye connection
            (3, 5), (4, 6),     # Ears-Shoulders
            (5, 6),             # Shoulder connection
            (5, 7), (7, 9),     # Left arm
            (6, 8), (8, 10),    # Right arm
            (5, 11), (6, 12),   # Shoulders-Pelvis
            (11, 12),           # Pelvis connection
            (11, 13), (13, 15), # Left leg
            (12, 14), (14, 16), # Right leg
        ]
        
        # Initialize utility classes
        self.keypoint_filter = KeypointFilter(self.original_keypoints)
        self.visualizer = PoseVisualizer(list(range(17)), self.skeleton_connections)
        self.preprocessor = ImagePreprocessor()
        self.depth_processor = DepthProcessor()
                
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model and check input/output information"""
        try:
            self.session = ort.InferenceSession(self.onnx_model_path)
            
            input_info = self.session.get_inputs()[0]
            self.input_name = input_info.name
            self.input_shape = input_info.shape
            
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            print(f"Model loaded: {os.path.basename(self.onnx_model_path)}")
            
            self.preprocessor.update_target_size(self.input_shape)
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise
    
    def process_single_image(self, image_path: str, depth_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Process a single image and depth data.
        
        Args:
            image_path (str): Path to the image file
            depth_path (str): Path to the depth file (bin file in meters)
            
        Returns:
            Tuple[np.ndarray, np.ndarray, Dict]: (keypoints_2D, keypoints_3D, distance_info)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image loading failed: {image_path}")
            return None, None, {}
        
        # Load depth data
        depth_image = self.depth_processor.read_depth_bin(depth_path)
        
        # Preprocessing
        input_tensor = self.preprocessor.preprocess_frame(image)
        
        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocessing (2D keypoints)
        keypoints_2d = self.keypoint_filter.postprocess_simcc_output(outputs, (image.shape[0], image.shape[1]))
        filtered_keypoints_2d = self.keypoint_filter.filter_keypoints(keypoints_2d)
        
        # Calculate 3D keypoints (in meters)
        keypoints_3d = self.depth_processor.get_keypoint_3d_coords(
            filtered_keypoints_2d, depth_image
        )
        
        # Calculate 3D distances between skeleton connections (in meters)
        distances = {}
        for start_idx, end_idx in self.skeleton_connections:
            if start_idx < len(keypoints_3d) and end_idx < len(keypoints_3d):
                distance = self.depth_processor.calculate_3d_distance(
                    keypoints_3d[start_idx], keypoints_3d[end_idx]
                )
                # Store all distances (both valid and invalid (-1))
                distances[(start_idx, end_idx)] = distance
        
        return filtered_keypoints_2d, keypoints_3d, distances
    
    def process_dataset(self, dataset_path: str, output_dir: str = "./output", details: bool = False):
        """
        Process the entire dataset.
        
        Args:
            dataset_path (str): Dataset directory path
            output_dir (str): Output directory path (default: ./output)
            details (bool): Whether to print detailed keypoint information
        """
        # Extract input directory name
        dataset_name = os.path.basename(os.path.normpath(dataset_path))
        
        # Dataset directory structure
        images_dir = os.path.join(dataset_path, "color_images")
        depth_dir = os.path.join(dataset_path, "bin")
        
        if not os.path.exists(images_dir):
            print(f"Image directory not found: {images_dir}")
            return
        
        if not os.path.exists(depth_dir):
            print(f"Depth data directory not found: {depth_dir}")
            return
        
        # Match files
        file_pairs = self.depth_processor.match_image_depth_files(images_dir, depth_dir)
        
        if len(file_pairs) == 0:
            print("No matching files found.")
            return
        
        # Create output directories
        results_dir = os.path.join(output_dir, f"{dataset_name}_result")
        os.makedirs(results_dir, exist_ok=True)
        
        # Create result_images directory for visualization results
        result_images_dir = os.path.join(results_dir, "result_images")
        os.makedirs(result_images_dir, exist_ok=True)
        
        # List for storing results
        all_results = []
        
        print(f"Processing started: {len(file_pairs)} files")
        
        for i, (image_path, depth_path) in enumerate(file_pairs):
            if (i + 1) % 10 == 0 or i == 0:  # Print every 10th or first
                print(f"   Progress: {i+1}/{len(file_pairs)}")
            
            # Process single image
            keypoints_2d, keypoints_3d, distances = self.process_single_image(image_path, depth_path)
            
            if keypoints_2d is not None:
                # Print detailed information (keypoint depths)
                if details:
                    self._print_single_image_keypoint_depths(i, os.path.basename(image_path), keypoints_3d, distances)
                
                # Load original image
                image = cv2.imread(image_path)
                
                # Visualization (display distances in centimeters)
                result_image = self.visualizer.draw_pose_with_distances(
                    image, keypoints_2d, distances, confidence_threshold=0.3, distance_unit="cm"
                )
                
                # Save result image in result_images directory
                result_filename = f"result_{i:04d}_{os.path.basename(image_path)}"
                result_path = os.path.join(result_images_dir, result_filename)
                cv2.imwrite(str(result_path), result_image)
                
                # Save result data
                result_data = {
                    "frame_id": i,
                    "image_path": image_path,
                    "depth_path": depth_path,
                    "keypoints_3d": keypoints_3d.tolist(),
                    "distances_meters": {f"{k[0]}-{k[1]}": v for k, v in distances.items() if v > 0},
                    "valid_keypoints": int(np.sum(keypoints_3d[:, 3] > 0))
                }
                all_results.append(result_data)
        
        # Save statistics to .out file
        analysis_output_path = os.path.join(results_dir, "analysis.out")
        self._save_results_and_statistics(all_results, results_dir, dataset_name, analysis_output_path)
        
        print(f"Processing complete! Results saved in '{results_dir}'")
        print(f"- Result images: {result_images_dir}")
        print(f"- Analysis results: {analysis_output_path}")

    def _save_results_and_statistics(self, results: List[Dict], results_dir: str, dataset_name: str, output_path: str):
        """Save results and print statistics"""
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
            
            self._print_distance_statistics_cm(results)
            self._print_keypoint_depth_statistics_cm(results)
            
        # Restore stdout
        sys.stdout = original_stdout
    
    def _calculate_trimmed_stats(self, data, trim_percent=5):
        """Calculate trimmed statistics (removing top and bottom 5%)"""
        if len(data) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        if len(data) == 1:
            return float(data[0]), 0.0, float(data[0]), float(data[0])
            
        sorted_data = np.sort(data)
        n = len(data)
        trim_size = int(n * trim_percent / 100)
        
        if trim_size * 2 >= n:  # If trimming would remove all data
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

    def _print_distance_statistics_cm(self, results: List[Dict]):
        """Print skeleton connection distance statistics and calculate MAE, RMSE, MAPE against ground truth"""
        # Ground truth distances (cm)
        ground_truth_cm = {
            "0-1": 4.0,   # Nose-Left Eye
            "0-2": 4.0,   # Nose-Right Eye
            "1-3": 7.0,   # Left Eye-Left Ear
            "2-4": 7.0,   # Right Eye-Right Ear
            "1-2": 8.0,   # Eye-Eye
            "3-5": 13.0,  # Left Ear-Left Shoulder
            "4-6": 13.0,  # Right Ear-Right Shoulder
            "5-6": 36.0,  # Shoulder-Shoulder
            "5-7": 28.0,  # Left Shoulder-Left Elbow
            "6-8": 28.0,  # Right Shoulder-Right Elbow
            "7-9": 25.0,  # Left Elbow-Left Wrist
            "8-10": 25.0, # Right Elbow-Right Wrist
            "5-11": 40.0, # Left Shoulder-Left Pelvis
            "6-12": 40.0, # Right Shoulder-Right Pelvis
            "11-12": 30.0,# Pelvis-Pelvis
            "11-13": 40.0,# Left Pelvis-Left Knee
            "12-14": 40.0,# Right Pelvis-Right Knee
            "13-15": 40.0,# Left Knee-Left Ankle
            "14-16": 40.0 # Right Knee-Right Ankle
        }
        
        all_distances = {}
        for result in results:
            for connection_key, distance in result['distances_meters'].items():
                if distance > 0:
                    if connection_key not in all_distances:
                        all_distances[connection_key] = []
                    all_distances[connection_key].append(distance * 100.0)  # Convert to cm
        
        print(f"\nSkeleton Connection Distance Statistics (centimeter units):")
        print("-" * 60)
        print("10% trimmed is applied")
        print()
        
        connection_names = {
            "0-1": "Nose-Left Eye", "0-2": "Nose-Right Eye",
            "1-3": "Left Eye-Left Ear", "2-4": "Right Eye-Right Ear",
            "1-2": "Eye-Eye", "3-5": "Left Ear-Left Shoulder", "4-6": "Right Ear-Right Shoulder",
            "5-6": "Shoulder-Shoulder", "5-7": "Left Shoulder-Left Elbow", "6-8": "Right Shoulder-Right Elbow",
            "7-9": "Left Elbow-Left Wrist", "8-10": "Right Elbow-Right Wrist",
            "5-11": "Left Shoulder-Left Pelvis", "6-12": "Right Shoulder-Right Pelvis",
            "11-12": "Pelvis-Pelvis", "11-13": "Left Pelvis-Left Knee", "12-14": "Right Pelvis-Right Knee",
            "13-15": "Left Knee-Left Ankle", "14-16": "Right Knee-Right Ankle"
        }
        
        # Storage for MAE, RMSE, MAPE
        mae_dict, rmse_dict, mape_dict = {}, {}, {}
        
        for connection_key in sorted(connection_names.keys()):
            distances = all_distances.get(connection_key, [])
            distances_array = np.array(distances) if distances else np.array([0.0])
            connection_label = connection_names.get(connection_key, connection_key)
            
            # Calculate trimmed statistics
            mean, std, min_val, max_val = self._calculate_trimmed_stats(distances_array)
            
            print(f"{connection_label}")
            print(f"   Average: {mean:.1f}cm")
            print(f"   Standard Deviation: {std:.1f}cm")
            print(f"   Range: {min_val:.1f}cm ~ {max_val:.1f}cm")
            print(f"   Data Count: {len(distances) if distances else 0}")
            
            # Calculate error against ground truth
            gt = ground_truth_cm.get(connection_key, 0.0)
            if len(distances_array) > 0:
                # Use trimmed data for error calculations
                trimmed_data = np.sort(distances_array)[int(len(distances_array)*0.05):int(len(distances_array)*0.95)]
                abs_err = np.abs(trimmed_data - gt)
                mae = np.mean(abs_err)
                rmse = np.sqrt(np.mean((trimmed_data - gt) ** 2))
                mape = np.mean(abs_err / gt * 100) if gt != 0 else 0.0
            else:
                mae = rmse = mape = 0.0
            
            print(f"   MAE: {mae:.2f}cm, RMSE: {rmse:.2f}cm, MAPE: {mape:.2f}%")
            mae_dict[connection_key] = mae
            rmse_dict[connection_key] = rmse
            mape_dict[connection_key] = mape
            print()
        
        # Print overall average error using trimmed means
        all_mae = np.mean(list(mae_dict.values()))
        all_rmse = np.mean(list(rmse_dict.values()))
        all_mape = np.mean(list(mape_dict.values()))
        print("Overall Connection Average Error:")
        print(f"   MAE: {all_mae:.2f}cm, RMSE: {all_rmse:.2f}cm, MAPE: {all_mape:.2f}%")

    def _print_keypoint_depth_statistics_cm(self, results: List[Dict]):
        """Print keypoint depth statistics (in centimeter units)"""
        keypoint_depths = {i: [] for i in range(17)}
        for result in results:
            keypoints_3d = np.array(result['keypoints_3d'])
            for i, (px, py, depth, conf) in enumerate(keypoints_3d):
                if conf > 0.3 and depth > 0:
                    keypoint_depths[i].append(depth * 100.0)  # Convert to cm
        
        print(f"Keypoint Depth Statistics (centimeter units):")
        print("-" * 60)
        print("10% trimmed is applied")
        print()
        
        for joint_id, depths in keypoint_depths.items():
            if depths:
                depths_array = np.array(depths)
                joint_name = self.keypoint_names.get(joint_id, f"Joint{joint_id}")
                
                # Calculate trimmed statistics
                mean, std, min_val, max_val = self._calculate_trimmed_stats(depths_array)
                
                print(f"{joint_name} (ID: {joint_id})")
                print(f"   Average Depth: {mean:.1f}cm")
                print(f"   Standard Deviation: {std:.1f}cm")
                print(f"   Range: {min_val:.1f}cm ~ {max_val:.1f}cm")
                print(f"   Detection Count: {len(depths_array)}")
                print()
    
    def _print_single_image_keypoint_depths(self, frame_id: int, image_file: str, keypoints_3d: np.ndarray, distances: Dict):
        """Print keypoint depths for a single image (in meter units)"""
        print(f"\nFrame {frame_id} ({image_file}):")
        print("-" * 40)
        
        # Keypoint depths
        print("Keypoint Depths:")
        for i, (px, py, depth, conf) in enumerate(keypoints_3d):
            if conf > 0.3:
                joint_name = self.keypoint_names.get(i, f"Joint{i}")
                print(f"   {joint_name}: {depth:.3f}m (Confidence: {conf:.2f})")
        
        # Print valid distances only
        valid_distances = {k: v for k, v in distances.items() if v > 0}
        if valid_distances:
            print("Connection Distances:")
            connection_names = {
                (0,1): "Nose-Left Eye", (0,2): "Nose-Right Eye", (1,3): "Left Eye-Left Ear", (2,4): "Right Eye-Right Ear",
                (1,2): "Eye-Eye", (3,5): "Left Ear-Left Shoulder", (4,6): "Right Ear-Right Shoulder",
                (5,6): "Shoulder-Shoulder", (5,7): "Left Shoulder-Left Elbow", (6,8): "Right Shoulder-Right Elbow",
                (7,9): "Left Elbow-Left Wrist", (8,10): "Right Elbow-Right Wrist",
                (5,11): "Left Shoulder-Left Pelvis", (6,12): "Right Shoulder-Right Pelvis",
                (11,12): "Pelvis-Pelvis", (11,13): "Left Pelvis-Left Knee", (12,14): "Right Pelvis-Right Knee",
                (13,15): "Left Knee-Left Ankle", (14,16): "Right Knee-Right Ankle"
            }
            
            for connection, distance in valid_distances.items():
                connection_label = connection_names.get(connection, f"{connection[0]}-{connection[1]}")
                print(f"   {connection_label}: {distance:.3f}m")

def main():
    parser = argparse.ArgumentParser(description='3D Pose Analysis')
    parser.add_argument('--model', required=True, help='ONNX model file path')
    parser.add_argument('--dataset', required=True, help='Dataset directory path')
    parser.add_argument('--output', default="./output", help='Output directory path (default: ./output)')
    parser.add_argument('--details', action='store_true', help='Print detailed keypoint information')
    
    args = parser.parse_args()
    
    # Initialize analysis system
    analyzer = ImageDepthPoseInference(args.model)
    
    # Process dataset
    analyzer.process_dataset(args.dataset, args.output, args.details)

if __name__ == "__main__":
    main() 