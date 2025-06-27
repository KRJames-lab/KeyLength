import numpy as np
from typing import List, Dict
from collections import defaultdict

class AnomalyDetector:
    """
    Detects and corrects anomalies in time-series data using Z-score with a sliding window.
    """
    def __init__(self, window_size: int, z_threshold: float = 2.0):
        """
        Initializes the AnomalyDetector.

        Args:
            window_size (int): The size of the sliding window (W). If 0, a global Z-score is used over all frames.
            z_threshold (float): The absolute Z-score value to determine an anomaly. Defaults to 2.0.
        """
        if window_size < -1:
            raise ValueError("Window size (W) must be -1, 0, or a positive integer.")
        self.window_size = window_size
        self.z_threshold = z_threshold

    def process(self, all_results: List[Dict]) -> List[Dict]:
        """
        Processes a list of inference results to detect and correct anomalies.

        Args:
            all_results (List[Dict]): The list of result dictionaries from inference.

        Returns:
            List[Dict]: The corrected list of result dictionaries.
        """
        if not all_results:
            return []

        connection_lengths = defaultdict(lambda: [None] * len(all_results))
        for i, result in enumerate(all_results):
            for key, distance in result.get('distances_meters', {}).items():
                connection_lengths[key][i] = distance * 100.0  # Work with cm

        corrected_lengths = {}
        for key, lengths in connection_lengths.items():
            corrected_lengths[key] = self._correct_series(lengths)

        corrected_all_results = []
        for i, original_result in enumerate(all_results):
            new_result = original_result.copy()
            new_distances = {}
            for key, distance in original_result.get('distances_meters', {}).items():
                # Use corrected value, converting back to meters
                new_distances[key] = corrected_lengths[key][i] / 100.0
            new_result['distances_meters'] = new_distances
            corrected_all_results.append(new_result)
            
        return corrected_all_results

    def _correct_series(self, series: List[float]) -> List[float]:
        """Applies anomaly detection and correction to a single time series."""
        
        if self.window_size == -1:
            # Calculate stats once from the entire series
            valid_values = [v for v in series if v is not None]
            if len(valid_values) < 2:
                # Not enough data to correct, return original series
                return list(series)
            
            global_mean = np.mean(valid_values)
            global_std = np.std(valid_values)
            
            corrected_series = list(series)
            if global_std == 0:
                # If all values are the same, fill Nones with the mean
                for i, val in enumerate(corrected_series):
                    if val is None:
                        corrected_series[i] = global_mean
                return corrected_series

            for i, val in enumerate(corrected_series):
                is_anomaly = False
                if val is None:
                    is_anomaly = True
                else:
                    z_score = (val - global_mean) / global_std
                    if abs(z_score) > self.z_threshold:
                        is_anomaly = True
                
                if is_anomaly:
                    corrected_series[i] = global_mean
            
            return corrected_series

        corrected_series = list(series)
        
        # Use cumulative stats if window_size is 0
        if self.window_size == 0:
            valid_so_far = []
            for i, val in enumerate(series):
                # If there's no valid data yet, try to populate it and continue
                if not valid_so_far:
                    if val is not None:
                        valid_so_far.append(val)
                    continue

                mean = np.mean(valid_so_far)
                
                is_anomaly = False

                if val is None:
                    is_anomaly = True

                elif len(valid_so_far) > 1:
                    std = np.std(valid_so_far)

                    if std > 0:
                        z_score = (val - mean) / std
                        if abs(z_score) > self.z_threshold:
                            is_anomaly = True
                
                if is_anomaly:
                    corrected_series[i] = mean
                    valid_so_far.append(mean)
                else:
                    valid_so_far.append(val)
        
        else: # Use sliding window
            for i in range(len(series)):
                start = max(0, i - self.window_size + 1)

                window = [x for x in series[start:i] if x is not None]

                if len(window) == 0:
                    continue

                mean = np.mean(window)

                is_anomaly = False
                if series[i] is None:
                    is_anomaly = True
                elif len(window) > 1:
                    std = np.std(window)
                    if std > 0:
                        z_score = (series[i] - mean) / std
                        if abs(z_score) > self.z_threshold:
                            is_anomaly = True
                
                if is_anomaly:
                    corrected_series[i] = mean
        return corrected_series 