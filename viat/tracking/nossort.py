#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    SingleNOCSORT: A self-contained single object tracker based on nOcSort algorithm
    Fully standalone implementation without relying on external libraries 
    except for basic NumPy and OpenCV packages.
"""

import numpy as np
import cv2
from copy import deepcopy

class KalmanFilter:
    """
    Simple Kalman filter implementation for tracking objects in XYSR format
    (x, y coordinates, aspect ratio, and scale)
    """
    def __init__(self):
        # State transition matrix (8x8)
        self.F = np.eye(8, 8)
        for i in range(4):
            self.F[i, i+4] = 1.0
        
        # Measurement matrix (4x8)
        self.H = np.zeros((4, 8))
        for i in range(4):
            self.H[i, i] = 1.0
        
        # Process noise covariance (8x8)
        self.Q = np.eye(8, 8) * 0.1
        
        # Measurement noise covariance (4x4)
        self.R = np.eye(4, 4) * 1.0
        
        # State covariance (8x8)
        self.P = np.eye(8, 8) * 10.0
        
        # State vector (8x1)
        self.x = np.zeros((8, 1))
        
        # Identity matrix
        self.I = np.eye(8, 8)
        
        # For steady state
        self.K_steady_state = None
        self.history_obs = []
        self.z = None
        self.x_post = None
        self.y = None
        
    def initialize(self, z):
        """Initialize the Kalman filter with a measurement z."""
        self.x = np.zeros((8, 1))
        self.x[0] = z[0]  # x position
        self.x[1] = z[1]  # y position
        self.x[2] = z[2]  # scale
        self.x[3] = z[3]  # aspect ratio
        self.P = np.eye(8, 8) * 10.0
        self.history_obs.append(z)
    
    def predict(self):
        """Predict the next state using the Kalman filter."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.x_post = self.x.copy()
        return self.x
    
    def update(self, z):
        """Update the Kalman filter with measurement z."""
        if z is None:
            self.history_obs.append(z)
            return
        
        # Calculate Kalman gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Update state estimate
        self.y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, self.y)
        
        # Update state covariance
        self.P = np.dot((self.I - np.dot(K, self.H)), self.P)
        
        # Save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        
        # Save history of observations
        self.history_obs.append(z)
    
    def compute_steady_state(self):
        """Compute the steady-state Kalman gain."""
        # Simplified steady-state computation
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        self.K_steady_state = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
    
    def update_steadystate(self, z):
        """Update using precomputed steady-state gain."""
        if z is None:
            self.history_obs.append(z)
            return
        
        # Calculate residual
        self.y = z - np.dot(self.H, self.x)
        
        # Update state estimate
        self.x = self.x + np.dot(self.K_steady_state, self.y)
        
        # Save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        
        # Save history of observations
        self.history_obs.append(z)

class STrack:
    """Single object tracking object with state information."""
    _next_id = 0
    
    def __init__(self, xyxy, score):
        # Convert [x1, y1, x2, y2] to [x, y, s, r] format
        # where s is scale (width) and r is aspect ratio (height/width)
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.score = score
        
        x1, y1, x2, y2 = xyxy
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        self.xysr = np.array([cx, cy, w, h/w])
        
        self.kf = KalmanFilter()
        self.kf.initialize(self.xysr)
        
        self.track_id = STrack._next_id
        STrack._next_id += 1
        
        self.is_activated = True
        self.time_since_update = 0
        self.max_age = 30  # Maximum frames to keep track without updates
        
        # For linear interpolation during GSI
        self.last_position = None
        self.current_position = xyxy
        
    def predict(self):
        """Predict next position using Kalman filter."""
        mean = self.kf.predict()[0:4]
        self.xysr = mean
        self.last_position = self.current_position.copy() if self.current_position is not None else None
        self.current_position = self.xysr_to_xyxy(self.xysr)
        return self.current_position
    
    def update(self, detection):
        """Update tracker with new detection."""
        self.time_since_update = 0
        self.xyxy = detection[0:4]
        self.score = detection[4]
        
        # Convert to xysr for Kalman filter
        x1, y1, x2, y2 = self.xyxy
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        self.xysr = np.array([cx, cy, w, h/w])
        
        self.kf.update(self.xysr)
        self.current_position = self.xyxy
        
    def xysr_to_xyxy(self, xysr):
        """Convert [x, y, s, r] to [x1, y1, x2, y2] format."""
        cx, cy, s, r = xysr
        w = s
        h = s * r
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return np.array([x1, y1, x2, y2])
    
    def get_state(self):
        """Return the current state as [x1, y1, x2, y2, score, id]."""
        return np.array([
            self.current_position[0],
            self.current_position[1],
            self.current_position[2],
            self.current_position[3],
            self.score,
            self.track_id
        ])

class NOCSORT:
    """Single object tracker based on nOcSort algorithm."""
    
    def __init__(self, future_frames=0, det_thresh=0.3, max_age=30):
        """Initialize tracker parameters."""
        self.max_age = max_age
        self.det_thresh = det_thresh
        self.future_frames = future_frames
        self.track = None
        self.frame_count = 0
        self.initialized = False
        
        # For Gaussian Smoothed Interpolation
        self.interpolation_interval = 20
        self.tau = 10.0  # Smoothing parameter
        self.results_history = []
        
    def reset(self):
        """Reset the tracker state."""
        self.track = None
        self.frame_count = 0
        self.initialized = False
        self.results_history = []
        STrack._next_id = 0
        
    def update(self, dets, img=None):
        """
        Update tracker with new detections.
        
        Args:
            dets: List of detections in format [x1, y1, x2, y2, score]
            img: Optional image for visualization
            
        Returns:
            List of tracking results in format [x1, y1, x2, y2, score, track_id]
        """
        self.frame_count += 1
        output_results = []
        
        # Initialize or update track with best detection
        if len(dets) > 0:
            # Get detection with highest confidence
            det_scores = np.array([d[4] for d in dets])
            best_idx = np.argmax(det_scores)
            best_det = dets[best_idx]
            
            if best_det[4] >= self.det_thresh:
                if not self.initialized:
                    # Initialize track
                    self.track = STrack(best_det[:4], best_det[4])
                    self.initialized = True
                else:
                    # Update existing track
                    self.track.update(best_det)
        
        # If we have an active track, predict its new state
        if self.track is not None:
            # Increment time since update if no detection matched
            if len(dets) == 0 or best_det[4] < self.det_thresh:
                self.track.time_since_update += 1
                
            # Predict new position
            self.track.predict()
            
            # Add track to results if it's still valid
            if self.track.time_since_update <= self.max_age:
                output_results.append(self.track.get_state())
            else:
                # Track is too old, remove it
                self.track = None
                self.initialized = False
        
        # Record results for potential GSI post-processing
        if len(output_results) > 0:
            # Add frame number to the beginning of each result
            for i in range(len(output_results)):
                result_with_frame = np.concatenate([[self.frame_count], output_results[i]])
                self.results_history.append(result_with_frame)
        
        return output_results
    
    def apply_gsi(self):
        """Apply Gaussian Smoothed Interpolation to tracking results."""
        if len(self.results_history) == 0:
            return []
        
        # Convert to numpy array
        results = np.array(self.results_history)
        
        # Apply linear interpolation
        interpolated_results = self._linear_interpolation(results, self.interpolation_interval)
        
        # For simplicity, we're returning just the interpolated results without Gaussian smoothing
        # A full implementation would include Gaussian Process Regression
        return interpolated_results[:, 1:]  # Remove frame number from output
    
    def _linear_interpolation(self, data, interval):
        """
        Apply linear interpolation between rows in the tracking results.
        First column is frame number, second is track ID.
        """
        # Sort data by frame
        sorted_data = data[np.argsort(data[:, 0])]
        result_rows = []
        previous_frame = None
        previous_row = None

        for row in sorted_data:
            current_frame = int(row[0])
            if previous_frame is not None and previous_frame + 1 < current_frame < previous_frame + interval:
                gap = current_frame - previous_frame - 1
                for i in range(1, gap + 1):
                    # Linear interpolation for each missing frame
                    interp_factor = i / (current_frame - previous_frame)
                    new_row = previous_row + (row - previous_row) * interp_factor
                    new_row[0] = previous_frame + i  # Set correct frame number
                    result_rows.append(new_row)
            result_rows.append(row)
            previous_frame, previous_row = current_frame, row

        return np.array(result_rows)
    
    def _iou_batch(self, bboxes1, bboxes2):
        """
        Calculate IoU between two sets of boxes (batch operation)
        """
        x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
        
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        
        interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
        
        boxAArea = (x12 - x11) * (y12 - y11)
        boxBArea = (x22 - x21) * (y22 - y21)
        
        iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
        return iou
