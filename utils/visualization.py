import cv2
import numpy as np
from typing import List, Tuple


class PoseVisualizer:
    """Responsible for pose visualization"""
    
    def __init__(self, selected_keypoints: List[int], skeleton_connections: List[Tuple[int, int]]):
        """
        Args:
            selected_keypoints (List[int]): List of keypoint indices to use
            skeleton_connections (List[Tuple[int, int]]): Skeleton connection relationships
        """
        self.selected_keypoints = selected_keypoints
        self.skeleton_connections = skeleton_connections
        
        # Create keypoint index mapping (original index -> filtered array index)
        self.keypoint_map = {orig_idx: i for i, orig_idx in enumerate(self.selected_keypoints)}
    
    def draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray, 
                     confidence_threshold: float = 0.3) -> np.ndarray:
        """
        Draw skeleton connection lines
        
        Args:
            frame (np.ndarray): Original frame
            keypoints (np.ndarray): Keypoint coordinates [num_keypoints, 3]
            confidence_threshold (float): Confidence threshold
            
        Returns:
            np.ndarray: Frame with skeleton drawn
        """
        result_frame = frame.copy()
        
        # Draw skeleton connection lines
        for start_idx, end_idx in self.skeleton_connections:
            if start_idx in self.keypoint_map and end_idx in self.keypoint_map:
                start_i = self.keypoint_map[start_idx]
                end_i = self.keypoint_map[end_idx]
                
                if (start_i < len(keypoints) and end_i < len(keypoints) and 
                    keypoints[start_i][2] > confidence_threshold and 
                    keypoints[end_i][2] > confidence_threshold):
                    
                    start_point = (int(keypoints[start_i][0]), int(keypoints[start_i][1]))
                    end_point = (int(keypoints[end_i][0]), int(keypoints[end_i][1]))
                    cv2.line(result_frame, start_point, end_point, (0, 0, 255), 1)
        
        return result_frame
    
    def draw_keypoints(self, frame: np.ndarray, keypoints: np.ndarray, 
                      confidence_threshold: float = 0.2, 
                      show_numbers: bool = True) -> np.ndarray:
        """
        Draw keypoints
        
        Args:
            frame (np.ndarray): Original frame
            keypoints (np.ndarray): Keypoint coordinates [num_keypoints, 3]
            confidence_threshold (float): Confidence threshold
            show_numbers (bool): Whether to show keypoint numbers
            
        Returns:
            np.ndarray: Frame with keypoints drawn
        """
        result_frame = frame.copy()
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > confidence_threshold:
                # Draw keypoint as a circle
                cv2.circle(result_frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                
                if show_numbers:
                    keypoint_idx = self.selected_keypoints[i] if i < len(self.selected_keypoints) else i
                    cv2.putText(result_frame, str(keypoint_idx), (int(x)+6, int(y)-6), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                    cv2.putText(result_frame, str(keypoint_idx), (int(x)+6, int(y)-6), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return result_frame
    
    def draw_pose(self, frame: np.ndarray, keypoints: np.ndarray, 
                 confidence_threshold: float = 0.3, 
                 show_numbers: bool = True) -> np.ndarray:
        """
        Draw entire pose (skeleton + keypoints)
        
        Args:
            frame (np.ndarray): Original frame
            keypoints (np.ndarray): Keypoint coordinates [num_keypoints, 3]
            confidence_threshold (float): Confidence threshold
            show_numbers (bool): Whether to show keypoint numbers
            
        Returns:
            np.ndarray: Frame with pose drawn
        """
        # Draw skeleton
        result_frame = self.draw_skeleton(frame, keypoints, confidence_threshold)
        
        # Draw keypoints
        result_frame = self.draw_keypoints(result_frame, keypoints, confidence_threshold, show_numbers)
        
        return result_frame
    
    def draw_skeleton_with_distances(self, frame: np.ndarray, keypoints: np.ndarray, 
                                     distances: dict, confidence_threshold: float = 0.3,
                                     distance_unit: str = "cm") -> np.ndarray:
        """
        Draw skeleton connection lines and 3D distance information (always in cm)
        """
        result_frame = frame.copy()
        for start_idx, end_idx in self.skeleton_connections:
            if start_idx in self.keypoint_map and end_idx in self.keypoint_map:
                start_i = self.keypoint_map[start_idx]
                end_i = self.keypoint_map[end_idx]
                if (start_i < len(keypoints) and end_i < len(keypoints) and 
                    keypoints[start_i][2] > confidence_threshold and 
                    keypoints[end_i][2] > confidence_threshold):
                    start_point = (int(keypoints[start_i][0]), int(keypoints[start_i][1]))
                    end_point = (int(keypoints[end_i][0]), int(keypoints[end_i][1]))
                    cv2.line(result_frame, start_point, end_point, (0, 0, 255), 1)
                    connection_key = (start_idx, end_idx)
                    reverse_key = (end_idx, start_idx)
                    distance = -1.0
                    if connection_key in distances:
                        distance = distances[connection_key]
                    elif reverse_key in distances:
                        distance = distances[reverse_key]
                    if distance > 0:
                        mid_x = (start_point[0] + end_point[0]) // 2
                        mid_y = (start_point[1] + end_point[1]) // 2
                        # Always convert to cm
                        display_distance = distance * 100.0
                        distance_text = f"{display_distance:.1f}cm"
                        text_size = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
                        cv2.rectangle(result_frame, 
                                    (mid_x - text_size[0]//2 - 2, mid_y - text_size[1] - 2),
                                    (mid_x + text_size[0]//2 + 2, mid_y + 2),
                                    (0, 0, 0), -1)
                        cv2.putText(result_frame, distance_text, 
                                  (mid_x - text_size[0]//2, mid_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return result_frame
    
    def draw_pose_with_distances(self, frame: np.ndarray, keypoints: np.ndarray, 
                                distances: dict, confidence_threshold: float = 0.3, 
                                show_numbers: bool = True, distance_unit: str = "cm") -> np.ndarray:
        """
        Draw entire pose and 3D distance information (always in cm)
        """
        result_frame = self.draw_skeleton_with_distances(frame, keypoints, distances, 
                                                        confidence_threshold, distance_unit)
        result_frame = self.draw_keypoints(result_frame, keypoints, confidence_threshold, show_numbers)
        return result_frame 