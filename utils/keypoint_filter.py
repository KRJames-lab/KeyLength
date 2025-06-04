import numpy as np
from typing import List, Tuple


class KeypointFilter:
    """Responsible for filtering and postprocessing keypoints"""
    
    def __init__(self, selected_keypoints: List[int]):
        """
        Args:
            selected_keypoints (List[int]): List of keypoint indices to use
        """
        self.selected_keypoints = selected_keypoints
    
    def filter_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Filter selected keypoints
        
        Args:
            keypoints (np.ndarray): Full keypoint array [num_keypoints, 3]
            
        Returns:
            np.ndarray: Filtered keypoint array
        """
        if len(keypoints) > max(self.selected_keypoints):
            filtered_keypoints = keypoints[self.selected_keypoints]
        else:
            # Return original if keypoints are insufficient
            filtered_keypoints = keypoints
        
        return filtered_keypoints
    
    def postprocess_simcc_output(self, outputs: List[np.ndarray], 
                                frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess SimCC output format
        
        Args:
            outputs (List[np.ndarray]): Model outputs
            frame_shape (Tuple[int, int]): Original frame size (height, width)
            
        Returns:
            np.ndarray: Keypoint coordinates [num_keypoints, 3] (x, y, confidence)
        """
        if len(outputs) >= 2:
            # If output format is SimCC (separate x and y coordinates)
            simcc_x = outputs[0]  # [batch, num_keypoints, width]
            simcc_y = outputs[1]  # [batch, num_keypoints, height]
            
            # Remove batch dimension
            simcc_x = simcc_x[0]
            simcc_y = simcc_y[0]
            
            # Find maximum position (argmax)
            x_coords = np.argmax(simcc_x, axis=1)
            y_coords = np.argmax(simcc_y, axis=1)
            
            # Calculate confidence (max value)
            x_confidence = np.max(simcc_x, axis=1)
            y_confidence = np.max(simcc_y, axis=1)
            confidence = (x_confidence + y_confidence) / 2
            
            # Scale to original image size
            model_width = simcc_x.shape[1]
            model_height = simcc_y.shape[1]
            
            x_coords = x_coords * (frame_shape[1] / model_width)
            y_coords = y_coords * (frame_shape[0] / model_height)
            
            # Create keypoint array [num_keypoints, 3]
            keypoints = np.stack([x_coords, y_coords, confidence], axis=1)
            
        else:
            # If output format is different, handle it
            output = outputs[0][0]  # Remove batch dimension
            if len(output.shape) == 2:  # [num_keypoints, 3]
                keypoints = output
            else:
                # Handle output format appropriately
                keypoints = output.reshape(-1, 3)
        
        return keypoints 