import cv2
import numpy as np

def calculate_frame_hash(frame):
    """
    Calculate a perceptual hash for an image frame using average hash (aHash).

    Args:
        frame (np.ndarray): Image frame (BGR or grayscale)

    Returns:
        str: Hexadecimal hash string
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    # Resize to 8x8
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    avg = small.mean()
    hash_bits = (small > avg).flatten()
    # Convert bits to hex string
    hash_str = ''.join(['1' if b else '0' for b in hash_bits])
    return '{:0>16x}'.format(int(hash_str, 2))

def mse_similarity(frame1, frame2):
    """
    Compute similarity between two frames using Mean Squared Error (MSE).

    Args:
        frame1 (np.ndarray): First image (BGR or grayscale)
        frame2 (np.ndarray): Second image (BGR or grayscale)

    Returns:
        float: Similarity score (1.0 = identical, 0.0 = maximally different)
    """
    # Convert to grayscale and resize for comparison
    if len(frame1.shape) == 3:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = frame1
    if len(frame2.shape) == 3:
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = frame2
    small1 = cv2.resize(gray1, (64, 64))
    small2 = cv2.resize(gray2, (64, 64))
    mse = np.mean((small1.astype("float") - small2.astype("float")) ** 2)
    similarity = 1 - (mse / 255**2)
    return similarity

def create_thumbnail(frame, size=(160, 90)):
    """
    Create a thumbnail image from a frame.

    Args:
        frame (np.ndarray): Image frame (BGR)
        size (tuple): Desired size (width, height)

    Returns:
        np.ndarray: Thumbnail image (RGB)
    """
    thumbnail = cv2.resize(frame, size)
    thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
    return thumbnail
