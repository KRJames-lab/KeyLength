#!/usr/bin/env python3
"""
RealSense bag file to extract aligned color image, depth image, depth matrix

Reference:
- https://github.com/IntelRealSense/librealsense/issues/4934
- https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
"""

import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
import os
from pathlib import Path

class RealSenseDataExtractor:
    def __init__(self, bag_file_path, output_dir="./extracted_data"):
        """
        Initialize RealSense data extractor
        
        Args:
            bag_file_path (str): path to bag file
            output_dir (str): directory to save extracted data
        """
        self.bag_file_path = bag_file_path
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.color_dir = self.output_dir / "color_images"
        self.depth_dir = self.output_dir / "depth_images"
        self.depth_matrix_dir = self.output_dir / "depth_bin"
        
        self._create_directories()
        
        # RealSense pipeline setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Alignment object (align depth to color)
        self.align_to_color = rs.align(rs.stream.color)
        
        # Colorizer (for visualization)
        self.colorizer = rs.colorizer()
        
    def _create_directories(self):
        """Create output directories."""
        for directory in [self.color_dir, self.depth_dir, self.depth_matrix_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        print(f"Output directories created: {self.output_dir}")
        
    def extract_data(self, visualize=False, save_colorized_depth=True):
        """
        Extract data from bag file.
        
        Args:
            visualize (bool): whether to visualize extraction process
            save_colorized_depth (bool): whether to save colorized depth image
        """
        try:
            # Enable streams in bag file
            self.config.enable_stream(rs.stream.color)
            self.config.enable_stream(rs.stream.depth)
            self.config.enable_device_from_file(self.bag_file_path, repeat_playback=False)
            
            # Start pipeline
            profile = self.pipeline.start(self.config)
            
            # Disable real-time playback (to avoid missing any frames)
            playback = profile.get_device().as_playback()
            playback.set_real_time(False)
            
            # Check total frames in bag file
            try:
                total_frames = playback.get_duration().total_seconds() * 30  # Rough estimate
                print(f"Estimated total frame count: {int(total_frames)} (estimate)")
            except:
                print("Unable to estimate total frame count.")
            
            # Get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            print(f"Depth Scale: {depth_scale}")
            
            frame_count = 0
            failed_attempts = 0
            max_failed_attempts = 3  # Allow consecutive failure attempts
            
            print("Starting data extraction...")
            
            while True:
                try:
                    # Wait for frame (increasing timeout)
                    frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                    
                    # Skip if frame is not fully prepared
                    if frames.size() < 2:
                        failed_attempts += 1
                        if failed_attempts > max_failed_attempts:
                            print(f"Failed to get frame {max_failed_attempts} times in a row. Exiting extraction.")
                            break
                        continue
                    
                    # Reset failed attempts if frame is successfully received
                    failed_attempts = 0
                        
                except RuntimeError as e:
                    failed_attempts += 1
                    print(f"Failed to wait for frame (attempt {failed_attempts}/{max_failed_attempts}): {str(e)}")
                    
                    if failed_attempts > max_failed_attempts:
                        print(f'Processed {frame_count} frames')
                        print(f'Total {frame_count} frames processed')
                        break
                    continue
                
                # Align depth to color
                aligned_frames = self.align_to_color.process(frames)
                
                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                # Frame validation
                if not aligned_depth_frame or not color_frame:
                    continue
                
                # Data conversion
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Depth matrix (actual distance values - in meters)
                depth_matrix_meters = depth_image * depth_scale
                
                # Convert color image to BGR (OpenCV format)
                color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                
                # File name generation
                frame_filename = f"frame_{frame_count:06d}"
                
                # 1. Save Color Image (PNG)
                color_path = self.color_dir / f"{frame_filename}.png"
                cv2.imwrite(str(color_path), color_image_bgr)
                
                # 2. Save Depth Matrix (BIN - actual distance values)
                depth_matrix_path = self.depth_matrix_dir / f"{frame_filename}.bin"
                # Convert to float32 to save file size
                depth_matrix_float32 = depth_matrix_meters.astype(np.float32)
                with open(str(depth_matrix_path), 'wb') as f:
                    f.write(depth_matrix_float32.tobytes())
                
                # 3. Save colorized depth image (optional)
                if save_colorized_depth:
                    depth_colormap = np.asanyarray(
                        self.colorizer.colorize(aligned_depth_frame).get_data()
                    )
                    depth_image_path = self.depth_dir / f"{frame_filename}.png"
                    cv2.imwrite(str(depth_image_path), depth_colormap)
                
                # Visualization (optional)
                if visualize:
                    self._visualize_frame(color_image_bgr, depth_colormap if save_colorized_depth else None)
                
                frame_count += 1
                
                if frame_count % 100 == 0:
                    print(f"Processed frames: {frame_count}")
                    
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            
        print(f"Extraction completed! Total {frame_count} frames saved in {self.output_dir}")
        
    def _visualize_frame(self, color_image, depth_colormap=None):
        """Visualize the frame."""
        if depth_colormap is not None:
            # Display color and depth images side by side
            images = np.hstack((color_image, depth_colormap))
            cv2.namedWindow('Color and Depth (Aligned)', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Color and Depth (Aligned)', images)
        else:
            cv2.namedWindow('Color Image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Color Image', color_image)
            
        # Press 'q' to stop visualization
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return False
        return True

def main():
    parser = argparse.ArgumentParser(
        description="Extract aligned color image, depth image, and depth matrix from RealSense bag file."
    )
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        required=True,
        help="Input bag file path"
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="./extracted_data",
        help="Output directory path (default: ./extracted_data)"
    )
    parser.add_argument(
        "-v", "--visualize", 
        action="store_true",
        help="Visualize extraction process"
    )
    parser.add_argument(
        "--no-colorized-depth", 
        action="store_true",
        help="Do not save colorized depth images"
    )
    
    args = parser.parse_args()
    
    # Input file validation
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        return
        
    if not args.input.endswith('.bag'):
        print(f"Error: Only .bag files are supported: {args.input}")
        return
    
    # Create and run data extractor
    extractor = RealSenseDataExtractor(args.input, args.output)
    extractor.extract_data(
        visualize=args.visualize,
        save_colorized_depth=not args.no_colorized_depth
    )

if __name__ == "__main__":
    main() 