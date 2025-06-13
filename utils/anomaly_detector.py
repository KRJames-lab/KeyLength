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
        if window_size < 0:
            raise ValueError("Window size (W) cannot be negative.")
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

        # 1. Restructure data: Group measurements by connection key
        connection_lengths = defaultdict(lambda: [None] * len(all_results))
        for i, result in enumerate(all_results):
            for key, distance in result.get('distances_meters', {}).items():
                connection_lengths[key][i] = distance * 100.0  # Work with cm

        # 2. Detect and correct anomalies for each connection
        corrected_lengths = {}
        for key, lengths in connection_lengths.items():
            corrected_lengths[key] = self._correct_series(lengths)

        # 3. Reconstruct all_results with corrected values
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
        corrected_series = list(series)
        
        # Use cumulative stats if window_size is 0
        if self.window_size == 0:
            valid_so_far = []
            for i, val in enumerate(series):
                prev_values = valid_so_far.copy()
                if len(prev_values) == 0:
                    continue
                mean = np.mean(prev_values)
                if val is None:
                    corrected_series[i] = mean
                    valid_so_far.append(mean)
                    continue
                if len(prev_values) == 1:
                    valid_so_far.append(val)
                    continue
                std = np.std(prev_values)
                if std == 0:
                    valid_so_far.append(val)
                    continue
                z_score = (val - mean) / std
                if abs(z_score) > self.z_threshold:
                    corrected_series[i] = mean
                    valid_so_far.append(mean)
                else:
                    valid_so_far.append(val)
        else: # Use sliding window
            for i in range(len(series)):
                start = max(0, i - self.window_size + 1)
                # 윈도우 내에서 현재 프레임(i)은 제외
                window = [x for x in series[start:i] if x is not None]

                if len(window) == 0:
                    continue

                mean = np.mean(window)
                if series[i] is None:
                    # If value is missing, treat as outlier and replace with mean of window
                    corrected_series[i] = mean
                    continue

                if len(window) == 1:
                    continue

                std = np.std(window)
                if std == 0:
                    continue

                z_score = (series[i] - mean) / std
                if abs(z_score) > self.z_threshold:
                    # mean을 대체값으로 사용
                    corrected_series[i] = mean
        return corrected_series 