import cv2
import numpy as np
from typing import Tuple


class ImagePreprocessor:
    """This class is responsible for image preprocessing"""
    
    def __init__(self, target_size: Tuple[int, int] = (192, 256)):
        """
        Args:
            target_size (Tuple[int, int]): Target image size (width, height)
        """
        self.target_width, self.target_height = target_size
        
        # COCO mean/standard deviation
        self.mean = np.array([123.0, 117.0, 104.0])
        self.std = np.array([58.0, 57.0, 57.0])
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame
        
        Args:
            frame (np.ndarray): Original frame (BGR)
            
        Returns:
            np.ndarray: Preprocessed input tensor
        """
        # Resize image
        resized_frame = cv2.resize(frame, (self.target_width, self.target_height))
        
        # BGR to RGB conversion
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Normalization
        normalized_frame = (rgb_frame - self.mean) / self.std
        
        # HWC to CHW conversion
        transposed_frame = normalized_frame.transpose(2, 0, 1)
        
        # Add batch dimension
        input_tensor = np.expand_dims(transposed_frame, axis=0).astype(np.float32)
        
        return input_tensor
    
    def update_target_size(self, input_shape: Tuple[int, ...]):
        """
        Update target size according to model input size
        
        Args:
            input_shape (Tuple[int, ...]): Model input shape
        """
        if len(input_shape) == 4:  # [batch, channel, height, width]
            self.target_height = input_shape[2] if input_shape[2] != -1 else 256
            self.target_width = input_shape[3] if input_shape[3] != -1 else 192 