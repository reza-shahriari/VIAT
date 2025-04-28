"""
Smart Edge Movement Module for Video Annotation Tool

This module provides computer vision-based edge detection and refinement
for more intelligent bounding box adjustments.
"""

import cv2
import numpy as np


def detect_edges(frame, rect, edge_type):
    """
    Detect edges in the region around the specified edge of the bounding box.

    Args:
        frame (numpy.ndarray): The current video frame
        rect (QRect): The current bounding box rectangle
        edge_type (str): Which edge to analyze ('top', 'bottom', 'left', 'right')

    Returns:
        int: Suggested position for the edge
    """
    # Convert QRect to coordinates
    x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()

    # Adjust padding based on object size - smaller objects need smaller padding
    # to avoid detecting unrelated edges
    size_factor = min(w, h)
    padding = max(
        5, min(20, int(size_factor * 0.2))
    )  # Between 5 and 20 pixels, scaled by object size

    # Define region of interest based on edge type
    if edge_type == "top":
        roi = frame[
            max(0, y - padding) : min(frame.shape[0], y + padding),
            max(0, x) : min(frame.shape[1], x + w),
        ]
        axis = 0  # y-axis
    elif edge_type == "bottom":
        roi = frame[
            max(0, y + h - padding) : min(frame.shape[0], y + h + padding),
            max(0, x) : min(frame.shape[1], x + w),
        ]
        axis = 0  # y-axis
    elif edge_type == "left":
        roi = frame[
            max(0, y) : min(frame.shape[0], y + h),
            max(0, x - padding) : min(frame.shape[1], x + padding),
        ]
        axis = 1  # x-axis
    elif edge_type == "right":
        roi = frame[
            max(0, y) : min(frame.shape[0], y + h),
            max(0, x + w - padding) : min(frame.shape[1], x + w + padding),
        ]
        axis = 1  # x-axis
    else:
        return None

    # Check if ROI is valid
    if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise - use smaller kernel for small objects
    kernel_size = 3 if min(roi.shape[0], roi.shape[1]) < 30 else 5
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Adjust Canny thresholds based on image content
    median = np.median(blurred)
    lower = int(max(0, (1.0 - 0.33) * median))
    upper = int(min(255, (1.0 + 0.33) * median))

    # Detect edges using Canny edge detector
    edges = cv2.Canny(blurred, lower, upper)

    # Find the strongest edge
    if axis == 0:  # horizontal edge (top/bottom)
        edge_strength = np.sum(edges, axis=1)
        if edge_type == "top":
            offset = np.argmax(edge_strength) - padding + y
        else:  # bottom
            offset = np.argmax(edge_strength) - padding + (y + h)
    else:  # vertical edge (left/right)
        edge_strength = np.sum(edges, axis=0)
        if edge_type == "left":
            offset = np.argmax(edge_strength) - padding + x
        else:  # right
            offset = np.argmax(edge_strength) - padding + (x + w)

    # Ensure the offset is within frame boundaries
    if axis == 0:  # y-axis
        offset = max(0, min(frame.shape[0] - 1, offset))
    else:  # x-axis
        offset = max(0, min(frame.shape[1] - 1, offset))

    return offset


def refine_edge_position(frame, rect, edge_type):
    """
    Refine the position of a bounding box edge using computer vision.

    Args:
        frame (numpy.ndarray): The current video frame
        rect (QRect): The current bounding box rectangle
        edge_type (str): Which edge to refine ('top', 'bottom', 'left', 'right')

    Returns:
        int: Refined position for the edge, or None if no refinement is possible
    """
    # First try edge detection
    edge_pos = detect_edges(frame, rect, edge_type)

    # If edge detection fails, try gradient-based approach
    if edge_pos is None:
        # Convert QRect to coordinates
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()

        # Adjust padding based on object size
        size_factor = min(w, h)
        padding = max(5, min(20, int(size_factor * 0.2)))

        if edge_type == "top":
            roi = frame[
                max(0, y - padding) : min(frame.shape[0], y + padding),
                max(0, x) : min(frame.shape[1], x + w),
            ]
            axis = 0  # y-axis
            current_pos = y
        elif edge_type == "bottom":
            roi = frame[
                max(0, y + h - padding) : min(frame.shape[0], y + h + padding),
                max(0, x) : min(frame.shape[1], x + w),
            ]
            axis = 0  # y-axis
            current_pos = y + h
        elif edge_type == "left":
            roi = frame[
                max(0, y) : min(frame.shape[0], y + h),
                max(0, x - padding) : min(frame.shape[1], x + padding),
            ]
            axis = 1  # x-axis
            current_pos = x
        elif edge_type == "right":
            roi = frame[
                max(0, y) : min(frame.shape[0], y + h),
                max(0, x + w - padding) : min(frame.shape[1], x + w + padding),
            ]
            axis = 1  # x-axis
            current_pos = x + w
        else:
            return None

        # Check if ROI is valid
        if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Calculate gradient magnitude - use smaller kernel for small objects
        kernel_size = 3
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

        # Find position of maximum gradient
        if axis == 0:  # horizontal edge (top/bottom)
            gradient_strength = np.sum(gradient_magnitude, axis=1)
            if edge_type == "top":
                offset = np.argmax(gradient_strength) - padding + y
            else:  # bottom
                offset = np.argmax(gradient_strength) - padding + (y + h)
        else:  # vertical edge (left/right)
            gradient_strength = np.sum(gradient_magnitude, axis=0)
            if edge_type == "left":
                offset = np.argmax(gradient_strength) - padding + x
            else:  # right
                offset = np.argmax(gradient_strength) - padding + (x + w)

        # Ensure the offset is within frame boundaries
        if axis == 0:  # y-axis
            offset = max(0, min(frame.shape[0] - 1, offset))
        else:  # x-axis
            offset = max(0, min(frame.shape[1] - 1, offset))

        # For small objects, be more conservative with edge adjustments
        # Only return the offset if it's within a reasonable range of current position
        max_shift = max(5, int(size_factor * 0.1))  # Scale max shift by object size
        if abs(offset - current_pos) > max_shift:
            # Limit the movement to max_shift
            if offset > current_pos:
                offset = current_pos + max_shift
            else:
                offset = current_pos - max_shift

        return offset

    return edge_pos


def smart_contour_detection(frame, rect, edge_type):
    """
    Use contour detection to find object boundaries for small objects.

    Args:
        frame (numpy.ndarray): The current video frame
        rect (QRect): The current bounding box rectangle
        edge_type (str): Which edge to refine ('top', 'bottom', 'left', 'right')

    Returns:
        int: Refined position for the edge, or None if no refinement is possible
    """
    # Convert QRect to coordinates
    x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()

    # For very small objects, expand the ROI to get more context
    padding = max(10, int(min(w, h) * 0.5))

    # Extract a region around the bounding box
    roi_x = max(0, x - padding)
    roi_y = max(0, y - padding)
    roi_w = min(frame.shape[1] - roi_x, w + 2 * padding)
    roi_h = min(frame.shape[0] - roi_y, h + 2 * padding)

    roi = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

    # Check if ROI is valid
    if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding for better segmentation of small objects
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Adjust contour coordinates to original frame coordinates
    for i in range(len(contours)):
        contours[i] = contours[i] + np.array([roi_x, roi_y])

    # Find the contour that best matches our current bounding box
    best_contour = None
    best_overlap = 0

    for contour in contours:
        # Get bounding rect of contour
        c_x, c_y, c_w, c_h = cv2.boundingRect(contour)

        # Calculate overlap with current rect
        overlap_x = max(0, min(x + w, c_x + c_w) - max(x, c_x))
        overlap_y = max(0, min(y + h, c_y + c_h) - max(y, c_y))
        overlap_area = overlap_x * overlap_y

        if overlap_area > best_overlap:
            best_overlap = overlap_area
            best_contour = contour

    if best_contour is None:
        return None

    # Get the bounding rect of the best contour
    c_x, c_y, c_w, c_h = cv2.boundingRect(best_contour)

    # Return the appropriate edge position
    if edge_type == "top":
        return c_y
    elif edge_type == "bottom":
        return c_y + c_h
    elif edge_type == "left":
        return c_x
    elif edge_type == "right":
        return c_x + c_w

    return None
