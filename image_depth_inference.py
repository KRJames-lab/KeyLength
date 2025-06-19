import cv2
import numpy as np
import onnxruntime as ort
import argparse
import os
from typing import Tuple, List, Dict
import sys

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
    GROUND_TRUTH_CM = {c["key"]: v for c, v in zip(COCO_CONNECTIONS, [26.3, 21.3, 21.3, 21.6, 21.2, 43.7, 43.5, 11.8, 30.7, 35.3, 31.6, 34.2])}

    DEPTH_TO_RGB_AFFINE_TRANSFORMS = dict(
        C001=np.array([[2.89310518e+00, -2.33353370e-02, 2.38200221e+02],
                       [1.14394588e-02, 2.88216964e+00, -3.67819523e+01],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        C002=np.array([[2.90778446e+00, -1.04633946e-02, 2.15505801e+02],
                       [-3.43830682e-03, 2.91094100e+00, -5.13416831e+01],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        C003=np.array([[2.89756295e+00, -7.16367761e-03, 2.12645813e+02],
                       [-1.26919485e-02, 2.89761514e+00, -6.53095423e+01],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    )

    def __init__(self, onnx_model_path: str, use_extension: bool = False, window_size: int = 5):
        """Initialize the 3D pose inference system.

        Args:
            onnx_model_path (str): ONNX model file path
            use_extension (bool): Flag to enable/disable the anomaly detection extension
            window_size (int): Window size for the anomaly detector
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
                
        self.use_extension = use_extension
        self.window_size = window_size
        
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
            depth_path (str): Path to the depth file (.bin or .png)
            
        Returns:
            Tuple[np.ndarray, np.ndarray, Dict]: (keypoints_2D, keypoints_3D, distance_info)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image loading failed: {image_path}")
            return None, None, {}

        # Load depth data based on file extension
        if depth_path.lower().endswith('.png'):
            # The camera ID is in the image filename, not the depth filename
            filename = os.path.basename(image_path)
            camera_id = None
            for cid in self.DEPTH_TO_RGB_AFFINE_TRANSFORMS.keys():
                if cid in filename:
                    camera_id = cid
                    break
            
            if camera_id is None:
                print(f"Could not determine camera ID from filename: {filename}. Skipping.")
                return None, None, {}

            affine_transform = self.DEPTH_TO_RGB_AFFINE_TRANSFORMS[camera_id]
            depth_image = self.depth_processor.read_depth_png(depth_path, affine_transform)
        else:
            # Assuming .bin file if not .png
            depth_image = self.depth_processor.read_depth_bin(depth_path)

        if depth_image is None or np.all(depth_image == 0):
            print(f"Depth data loading or processing failed for: {depth_path}")
            return None, None, {}
        
        # Preprocessing
        input_tensor = self.preprocessor.preprocess_frame(image)
        
        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocessing (2D keypoints)
        keypoints_2d_all_17 = self.keypoint_filter.postprocess_simcc_output(outputs, (image.shape[0], image.shape[1]))
        
        # --- DEBUG START ---
        print(f"DEBUG: Image: {os.path.basename(image_path)}")
        print(f"DEBUG: keypoints_2d_all_17 (x, y, conf):\n{keypoints_2d_all_17}")
        # --- DEBUG END ---
        
        # Filter to keep only the body keypoints (indices 5-16)
        keypoints_2d = keypoints_2d_all_17[self.keypoints_to_use_indices]
        
        # --- DEBUG START ---
        print(f"DEBUG: keypoints_2d (12 body keypoints, x, y, conf):\n{keypoints_2d}")
        # --- DEBUG END ---
        
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
    
    def process_dataset(self, dataset_path: str, output_dir: str = "./output", png_depth_dir: str = None, details: bool = False, trim_ratio: float = 0.1):
        """
        Process the entire dataset.
        
        Args:
            dataset_path (str): Dataset directory path
            output_dir (str): Output directory path (default: ./output)
            png_depth_dir (str, optional): Path to the root directory for PNG depth files. Defaults to None.
            details (bool): Whether to print detailed keypoint information
            trim_ratio (float): Ratio to trim from each end for statistics (e.g., 0.1 for 10%) (default: 0.1)
        """
        # Extract input directory name
        dataset_name = os.path.basename(os.path.normpath(dataset_path))
        
        # Dataset directory structure
        images_dir = os.path.join(dataset_path, "color_images")
        
        if not os.path.exists(images_dir):
            print(f"Image directory not found: {images_dir}")
            return

        if png_depth_dir:
            if not os.path.exists(png_depth_dir):
                print(f"PNG depth directory not found: {png_depth_dir}")
                return
            file_pairs = self.depth_processor.match_image_png_depth_files(images_dir, png_depth_dir)
        else:
            depth_dir = os.path.join(dataset_path, "bin")
            if not os.path.exists(depth_dir):
                print(f"Depth data directory not found: {depth_dir}")
                return
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

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="3D Pose Inference from Image and Depth Data")
    parser.add_argument("--model", type=str, default="./onnx/model.onnx",
                        help="ONNX model file path")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset directory path")
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory path (default: ./output)")
    parser.add_argument("--png_depth_dir", type=str, default=None,
                        help="Root directory for PNG depth files. If provided, this will be used instead of the 'bin' directory.")
    parser.add_argument("--details", action="store_true",
                        help="Print detailed keypoint depth information for each frame")
    parser.add_argument("--trim_ratio", type=float, default=0.1,
                        help="Ratio to trim from each end of the data for statistical analysis (e.g., 0.1 for 10%)")
    parser.add_argument("--use_extension", action="store_true", 
                        help="Enable the anomaly detection and correction extension")
    parser.add_argument("--window_size", type=int, default=5, 
                        help="Window size for the anomaly detector extension")

    args = parser.parse_args()
    
    # Initialize the inference system with extension flag
    try:
        inference_system = ImageDepthPoseInference(
            onnx_model_path=args.model,
            use_extension=args.use_extension,
            window_size=args.window_size
        )
    except Exception as e:
        print(f"Initialization failed: {e}")
        sys.exit(1)

    # Process the entire dataset
    inference_system.process_dataset(
        dataset_path=args.dataset,
        output_dir=args.output,
        png_depth_dir=args.png_depth_dir,
        details=args.details,
        trim_ratio=args.trim_ratio
    )

if __name__ == "__main__":
    main() 