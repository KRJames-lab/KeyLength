import numpy as np
import os
from typing import Tuple, List, Dict
import cv2


class DepthProcessor:
    """Depth data processing and 3D coordinate calculation"""
    
    def __init__(self):
        """Initialization (actual camera parameter usage)"""
        # Actual RGB camera internal parameters
        self.fx = 596.901611328125  # Actual focal length x
        self.fy = 597.372497558594  # Actual focal length y
        self.cx = 314.991790771484  # Actual principal point x (PPX)
        self.cy = 245.886520385742  # Actual principal point y (PPY)
        
        # Lens distortion coefficients (Brown-Conrady model)
        self.distortion_coeffs = np.array([
            0.164545848965645,      # k1
            -0.503093838691711,     # k2
            -0.000860264350194484,  # p1
            -0.000300821207929403,  # p2
            0.462810575962067       # k3
        ])
        
        # Camera matrix generation
        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
    
    def read_depth_bin(self, bin_path: str, width: int = 640, height: int = 480) -> np.ndarray:
        """
        Read binary depth file (actual distance value in meters)
        
        Args:
            bin_path (str): Binary file path
            width (int): Image width (default: 640)
            height (int): Image height (default: 480)
            
        Returns:
            np.ndarray: Depth data [height, width] (meters)
        """
        try:
            with open(bin_path, 'rb') as f:
                data = f.read()
                
            # Read entire data as 32bit float
            all_data = np.frombuffer(data, dtype=np.float32)
            
            # Calculate depth pixel count
            depth_pixels = width * height
            
            if len(all_data) == depth_pixels:
                depth_data = all_data
            else:
                raise ValueError(f"Depth data size mismatch: {bin_path} (expected: {depth_pixels}, actual: {len(all_data)})")
            
            # Reshape to 640x480
            depth_image = depth_data.reshape((height, width))
            
            return depth_image.astype(np.float32)
            
        except Exception as e:
            print(f"Depth file read failed: {bin_path}, error: {e}")
            return np.zeros((height, width), dtype=np.float32)
    
    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Correct lens distortion of 2D keypoints
        
        Args:
            points (np.ndarray): 2D keypoint coordinates [N, 2] (x, y)
            
        Returns:
            np.ndarray: Corrected 2D coordinates [N, 2]
        """
        # Reconstruct keypoints [N, 1, 2] format
        points_reshaped = points.reshape(-1, 1, 2)
        
        # Correct lens distortion
        undistorted_points = cv2.undistortPoints(
            points_reshaped,
            self.camera_matrix,
            self.distortion_coeffs,
            P=self.camera_matrix  # Return result as pixel coordinates
        )
        
        return undistorted_points.reshape(-1, 2)
    
    def get_keypoint_3d_coords(self, keypoints_2d: np.ndarray, depth_image: np.ndarray) -> np.ndarray:
        """
        Convert 2D keypoints to 3D coordinates (applying lens distortion correction)
                
        Args:
            keypoints_2d (np.ndarray): 2D keypoints [num_keypoints, 3] (x, y, confidence)
            depth_image (np.ndarray): Depth image [height, width] (meters)
            
        Returns:
            np.ndarray: 3D keypoints [num_keypoints, 4] (x, y, z, confidence)
        """
        keypoints_3d = []
        
        # Select valid keypoints only
        valid_mask = keypoints_2d[:, 2] > 0
        valid_points = keypoints_2d[valid_mask, :2]
        
        if len(valid_points) > 0:
            # Correct lens distortion
            undistorted_points = self.undistort_points(valid_points)
            
            # Result processing
            idx = 0
            for i in range(len(keypoints_2d)):
                if valid_mask[i]:
                    x_2d, y_2d = undistorted_points[idx]
                    idx += 1
                    
                    # Convert pixel coordinates to integers
                    u = int(round(x_2d))
                    v = int(round(y_2d))
                    
                    # Check image range
                    if 0 <= u < depth_image.shape[1] and 0 <= v < depth_image.shape[0]:
                        depth_meters = depth_image[v, u]
                        
                        if depth_meters > 0:
                            # Calculate actual 3D coordinates (using corrected coordinates)
                            x_3d = (x_2d - self.cx) * depth_meters / self.fx
                            y_3d = (y_2d - self.cy) * depth_meters / self.fy
                            z_3d = depth_meters
                        else:
                            x_3d, y_3d, z_3d = 0.0, 0.0, 0.0
                    else:
                        x_3d, y_3d, z_3d = 0.0, 0.0, 0.0
                else:
                    x_3d, y_3d, z_3d = 0.0, 0.0, 0.0
                
                keypoints_3d.append([x_3d, y_3d, z_3d, keypoints_2d[i, 2]])
        
        return np.array(keypoints_3d)
    
    def calculate_3d_distance(self, point1_3d: np.ndarray, point2_3d: np.ndarray) -> float:
        """
        Calculate actual distance between two 3D points (in meters)
        
        Args:
            point1_3d (np.ndarray): First 3D point [x_3d, y_3d, z_3d, conf]
            point2_3d (np.ndarray): Second 3D point [x_3d, y_3d, z_3d, conf]
            
        Returns:
            float: 3D distance (meters), return -1 if depth value is 0
        """
        if point1_3d[3] <= 0 or point2_3d[3] <= 0:  # confidence check
            return -1.0
        
        if point1_3d[2] <= 0 or point2_3d[2] <= 0:  # depth check
            return -1.0
        
        # Use already converted 3D coordinates (from get_keypoint_3d_coords)
        x1_real = point1_3d[0]
        y1_real = point1_3d[1]
        z1_real = point1_3d[2]
        
        x2_real = point2_3d[0]
        y2_real = point2_3d[1]
        z2_real = point2_3d[2]
        
        # Calculate Euclidean distance (in meters)
        dx = x1_real - x2_real
        dy = y1_real - y2_real
        dz = z1_real - z2_real
        
        distance = np.sqrt(dx*dx + dy*dy + dz*dz)
        return distance
    
    def match_image_depth_files(self, images_dir: str, depth_dir: str) -> List[Tuple[str, str]]:
        """
        Match images and depth files in chronological order
        
        Args:
            images_dir (str): Image directory path
            depth_dir (str): Depth file directory path
            
        Returns:
            List[Tuple[str, str]]: List of (image_path, depth_path) pairs
        """
        # Get image file list
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend([f for f in os.listdir(images_dir) if f.lower().endswith(ext)])
        image_files.sort()
        
        # Get depth file list
        depth_files = [f for f in os.listdir(depth_dir) if f.endswith('.bin')]
        depth_files.sort()
        
        # Match file count (match with the smaller one)
        min_count = min(len(image_files), len(depth_files))
        
        matched_pairs = []
        for i in range(min_count):
            image_path = os.path.join(images_dir, image_files[i])
            depth_path = os.path.join(depth_dir, depth_files[i])
            matched_pairs.append((image_path, depth_path))
        
        print(f"Matched files: {len(matched_pairs)}")
        
        return matched_pairs