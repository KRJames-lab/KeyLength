import cv2
import numpy as np
import onnxruntime as ort
import argparse
import os
from typing import Tuple, List, Dict
import sys
import json
from pathlib import Path

from utils import PoseVisualizer, KeypointFilter, ImagePreprocessor, DepthProcessor, Analysis, AnomalyDetector
from utils.visualization_analysis import save_boxplot, save_histogram, save_framewise_plot

class ImageDepthPoseInference:
    # Connections and ground truth data have been updated to exclude face keypoints (0-4)
    # and re-index the remaining keypoints from 0 to 11.
    COCO_CONNECTIONS = [
        {"key": "0-1", "label": "Shoulder connection", "group": "upper_body"},
        {"key": "0-2", "label": "Left Shoulder-Left Elbow", "group": "upper_body"},
        {"key": "2-4", "label": "Left Elbow-Left Wrist", "group": "upper_body"},
        {"key": "1-3", "label": "Right Shoulder-Right Elbow", "group": "upper_body"},
        {"key": "3-5", "label": "Right Elbow-Right Wrist", "group": "upper_body"},
        {"key": "0-6", "label": "Left Shoulder-Left Pelvis", "group": "upper_body"},
        {"key": "1-7", "label": "Right Shoulder-Right Pelvis", "group": "upper_body"},
        {"key": "6-7", "label": "Pelvis connection", "group": "upper_body"},
        {"key": "6-8", "label": "Left Pelvis-Left Knee", "group": "lower_body"},
        {"key": "8-10", "label": "Left Knee-Left Ankle", "group": "lower_body"},
        {"key": "7-9", "label": "Right Pelvis-Right Knee", "group": "lower_body"},
        {"key": "9-11", "label": "Right Knee-Right Ankle", "group": "lower_body"},
    ]
    GROUND_TRUTH_CM = {c["key"]: v for c, v in zip(COCO_CONNECTIONS, [36.0, 28.0, 25.0, 28.0, 25.0, 40.0, 40.0, 30.0, 40.0, 40.0, 40.0, 40.0])}

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
        
        # Original 17 keypoints from the COCO model output
        self.original_keypoints_indices = list(range(17))

        # We will now exclude the face keypoints (0-4)
        self.keypoints_to_use_indices = list(range(5, 17))
        
        # Create a mapping from original index (5-16) to new index (0-11)
        self.original_to_new_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(self.keypoints_to_use_indices)}

        # New keypoint names, re-indexed from 0
        self.keypoint_names = {
            self.original_to_new_map[5]: "Left Shoulder", self.original_to_new_map[6]: "Right Shoulder",
            self.original_to_new_map[7]: "Left Elbow", self.original_to_new_map[8]: "Right Elbow",
            self.original_to_new_map[9]: "Left Wrist", self.original_to_new_map[10]: "Right Wrist",
            self.original_to_new_map[11]: "Left Pelvis", self.original_to_new_map[12]: "Right Pelvis",
            self.original_to_new_map[13]: "Left Knee", self.original_to_new_map[14]: "Right Knee",
            self.original_to_new_map[15]: "Left Ankle", self.original_to_new_map[16]: "Right Ankle"
        }
        
        # Original skeleton connections (using original 0-16 indices)
        original_skeleton = [
            (5, 6),             # Shoulder connection
            (5, 7), (7, 9),     # Left arm
            (6, 8), (8, 10),    # Right arm
            (5, 11), (6, 12),   # Shoulders-Pelvis
            (11, 12),           # Pelvis connection
            (11, 13), (13, 15), # Left leg
            (12, 14), (14, 16), # Right leg
        ]

        # Remap skeleton connections to new 0-11 indices
        self.skeleton_connections = [
            (self.original_to_new_map[start], self.original_to_new_map[end])
            for start, end in original_skeleton
        ]
        
        # Initialize utility classes with the new 12-keypoint configuration
        self.keypoint_filter = KeypointFilter(self.original_keypoints_indices)
        self.visualizer = PoseVisualizer(list(self.keypoint_names.keys()), self.skeleton_connections)
        self.preprocessor = ImagePreprocessor()
        self.depth_processor = DepthProcessor()
        self.reporter = Analysis(
            self.COCO_CONNECTIONS, self.GROUND_TRUTH_CM, self.keypoint_names
        )
                
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
        keypoints_2d_all_17 = self.keypoint_filter.postprocess_simcc_output(outputs, (image.shape[0], image.shape[1]))
        
        # Filter to keep only the body keypoints (indices 5-16)
        keypoints_2d = keypoints_2d_all_17[self.keypoints_to_use_indices]
        
        # Calculate 3D keypoints (in meters) for the 12 body keypoints
        keypoints_3d = self.depth_processor.get_keypoint_3d_coords(
            keypoints_2d, depth_image
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
        
        return keypoints_2d, keypoints_3d, distances
    
    def process_dataset(self, dataset_path: str, output_dir: str = "./output", details: bool = False, trim_ratio: float = 0.1):
        """
        Process the entire dataset.
        
        Args:
            dataset_path (str): Dataset directory path
            output_dir (str): Output directory path (default: ./output)
            details (bool): Whether to print detailed keypoint information
            trim_ratio (float): Ratio to trim from each end for statistics (e.g., 0.1 for 10%) (default: 0.1)
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
        
        # Anomaly Detection and Correction
        if self.use_extension:
            print(f"Applying anomaly detection with window size W={self.window_size}...")
            detector = AnomalyDetector(window_size=self.window_size)
            all_results = detector.process(all_results)
            print("Anomaly detection complete.")

        # Save statistics to .out file
        analysis_output_path = os.path.join(results_dir, "analysis.out")
        
        # Use AnalysisReporter to generate the report
        self.reporter.save_results_and_statistics(
            all_results, results_dir, dataset_name, analysis_output_path, trim_ratio
        )
        
        print(f"Processing complete! Results saved in '{results_dir}'")
        print(f"- Result images: {result_images_dir}")
        print(f"- Analysis results: {analysis_output_path}")

    def _print_single_image_keypoint_depths(self, frame_id: int, image_file: str, keypoints_3d: np.ndarray, distances: Dict):
        """Print detailed information for a single image"""
        print(f"\nFrame {frame_id} ({image_file}):")
        print("-" * 40)
        print("Keypoint Depths:")
        for i, (x, y, depth, conf) in enumerate(keypoints_3d):
            if conf > 0.3:
                keypoint_name = self.keypoint_names.get(i, f"Unknown {i}")
                print(f"   {keypoint_name}: {depth:.3f}m (Confidence: {conf:.2f})")
        
        print("Connection Distances:")
        for (start_idx, end_idx), dist in distances.items():
            if dist > 0:
                start_name = self.keypoint_names.get(start_idx, f"KP{start_idx}")
                end_name = self.keypoint_names.get(end_idx, f"KP{end_idx}")
                print(f"   {start_name}-{end_name}: {dist:.3f}m")

    def match_image_json_files(self, image_dir: str, json_dir: str) -> List[Tuple[str, str]]:
        """Matches image files with json files based on filename."""
        image_files = sorted(list(Path(image_dir).glob('*')))
        json_files = {f.stem.split('_rgb')[0]: f for f in Path(json_dir).glob('*.json')}
        
        pairs = []
        for img_file in image_files:
            img_stem = img_file.stem.split('_rgb')[0]
            if img_stem in json_files:
                pairs.append((str(img_file), str(json_files[img_stem])))
        return pairs

    def process_single_image_with_json(self, image_path: str, json_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Process a single image using 3D keypoints from a JSON file.
        2D keypoints for visualization are still inferred from the image.
        
        Args:
            image_path (str): Path to the image file
            json_path (str): Path to the JSON file with 3D keypoints
            
        Returns:
            Tuple[np.ndarray, np.ndarray, Dict]: (keypoints_2D, keypoints_3D, distance_info)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image loading failed: {image_path}")
            return None, None, {}

        # Load 3D keypoints from JSON
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            # Assuming the first person's data is used if multiple people are present
            pose_keypoints_3d_flat = data['people'][0]['pose_keypoints_3d']
            
            # The JSON seems to contain 12 keypoints (x,y,z)
            keypoints_3d_xyz = np.array(pose_keypoints_3d_flat).reshape(-1, 3)
            
            # Create a (N, 4) array with confidence
            keypoints_3d = np.zeros((keypoints_3d_xyz.shape[0], 4))
            keypoints_3d[:, :3] = keypoints_3d_xyz
            
            # Set confidence to 1 for non-zero keypoints
            valid_kps = np.any(keypoints_3d_xyz != 0, axis=1)
            keypoints_3d[valid_kps, 3] = 1.0

            # Ensure we have 12 keypoints as expected by the skeleton
            if keypoints_3d.shape[0] != 12:
                 print(f"Warning: Expected 12 keypoints, but found {keypoints_3d.shape[0]} in {json_path}. Padding/truncating.")
                 new_kps = np.zeros((12, 4))
                 n_copy = min(12, keypoints_3d.shape[0])
                 new_kps[:n_copy] = keypoints_3d[:n_copy]
                 keypoints_3d = new_kps

        except (IOError, json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Failed to load/parse 3D keypoints from {json_path}: {e}")
            return None, None, {}

        # For visualization, we still need 2D keypoints. We'll infer them from the image.
        input_tensor = self.preprocessor.preprocess_frame(image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        keypoints_2d_all_17 = self.keypoint_filter.postprocess_simcc_output(outputs, (image.shape[0], image.shape[1]))
        keypoints_2d = keypoints_2d_all_17[self.keypoints_to_use_indices]

        # Calculate 3D distances from JSON keypoints (in meters)
        distances = {}
        for start_idx, end_idx in self.skeleton_connections:
            if start_idx < len(keypoints_3d) and end_idx < len(keypoints_3d):
                distance = self.depth_processor.calculate_3d_distance(
                    keypoints_3d[start_idx], keypoints_3d[end_idx]
                )
                distances[(start_idx, end_idx)] = distance
        
        return keypoints_2d, keypoints_3d, distances

    def process_dataset_with_json(self, dataset_path: str, json_dir: str, output_dir: str = "./output", details: bool = False, trim_ratio: float = 0.1):
        """
        Process the entire dataset using images and JSON files for 3D keypoints.
        """
        dataset_name = os.path.basename(os.path.normpath(dataset_path))
        images_dir = os.path.join(dataset_path, "color_images")

        if not os.path.exists(images_dir):
            print(f"Image directory not found: {images_dir}")
            return
        
        if not os.path.exists(json_dir):
            print(f"JSON directory not found: {json_dir}")
            return

        file_pairs = self.match_image_json_files(images_dir, json_dir)
        
        if len(file_pairs) == 0:
            print("No matching image/JSON files found.")
            return
        
        results_dir = os.path.join(output_dir, f"{dataset_name}_json_result")
        os.makedirs(results_dir, exist_ok=True)
        
        result_images_dir = os.path.join(results_dir, "result_images")
        os.makedirs(result_images_dir, exist_ok=True)
        
        all_results = []
        
        print(f"Processing started: {len(file_pairs)} files using JSON 3D keypoints")
        
        for i, (image_path, json_path) in enumerate(file_pairs):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"   Progress: {i+1}/{len(file_pairs)}")
            
            keypoints_2d, keypoints_3d, distances = self.process_single_image_with_json(image_path, json_path)
            
            if keypoints_2d is not None:
                if details:
                    self._print_single_image_keypoint_depths(i, os.path.basename(image_path), keypoints_3d, distances)
                
                image = cv2.imread(image_path)
                
                result_image = self.visualizer.draw_pose_with_distances(
                    image, keypoints_2d, distances, confidence_threshold=0.3, distance_unit="cm"
                )
                
                result_filename = f"result_{i:04d}_{os.path.basename(image_path)}"
                result_path = os.path.join(result_images_dir, result_filename)
                cv2.imwrite(str(result_path), result_image)
                
                result_data = {
                    "frame_id": i,
                    "image_path": image_path,
                    "json_path": json_path,
                    "keypoints_3d": keypoints_3d.tolist(),
                    "distances_meters": {f"{k[0]}-{k[1]}": v for k, v in distances.items() if v > 0},
                    "valid_keypoints": int(np.sum(keypoints_3d[:, 3] > 0))
                }
                all_results.append(result_data)
        
        if self.use_extension:
            print(f"Applying anomaly detection with window size W={self.window_size}...")
            detector = AnomalyDetector(window_size=self.window_size)
            all_results = detector.process(all_results)
            print("Anomaly detection complete.")

        analysis_output_path = os.path.join(results_dir, "analysis.out")
        
        self.reporter.save_results_and_statistics(
            all_results, results_dir, dataset_name, analysis_output_path, trim_ratio
        )
        
        print(f"Processing complete! Results saved in '{results_dir}'")
        print(f"- Result images: {result_images_dir}")
        print(f"- Analysis results: {analysis_output_path}")

def main():
    parser = argparse.ArgumentParser(description='3D Pose Analysis')
    parser.add_argument('--model', required=True, help='ONNX model file path')
    parser.add_argument('--dataset', help='Dataset directory path, containing color_images folder.')
    parser.add_argument('--json_dir', help='Directory of JSON files with 3D keypoints.')
    parser.add_argument('--output', default="./output", help='Output directory path (default: ./output)')
    parser.add_argument('--details', action='store_true', help='Print detailed keypoint information')
    parser.add_argument('--trim', type=float, default=0.1, help='Ratio to trim from each end for statistics (e.g., 0.1 for 10%%) (default: 0.1)')
    parser.add_argument('--extension', action='store_true', help='Enable anomaly detection and correction extension.')
    parser.add_argument('--W', type=int, default=30, help='Sliding window size for anomaly detection (W=0 for all frames).')
    
    args = parser.parse_args()
    
    if not args.dataset:
        parser.error("--dataset is a required argument.")

    # Initialize analysis system
    analyzer = ImageDepthPoseInference(args.model)
    
    # Pass extension args to the processor
    analyzer.use_extension = args.extension
    analyzer.window_size = args.W

    # Process dataset
    if args.json_dir:
        print("Running in JSON mode.")
        analyzer.process_dataset_with_json(args.dataset, args.json_dir, args.output, args.details, args.trim)
    else:
        print("Running in depth image mode.")
        analyzer.process_dataset(args.dataset, args.output, args.details, args.trim)

if __name__ == "__main__":
    main() 