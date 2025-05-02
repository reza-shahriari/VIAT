"""
Video Manager for VIAT

This module contains the VideoManager class which handles all video-related
functionality for the Video Image Annotation Tool.
"""

import os
import cv2
import json
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from PyQt5.QtCore import QTimer, QObject, pyqtSignal


class VideoManager(QObject):
    """
    Manages video loading, playback, and frame handling for the VIAT application.
    
    This class encapsulates all video-related functionality, including:
    - Loading video files
    - Frame navigation
    - Playback control
    - Frame duplication detection
    - Video metadata handling
    """
    
    # Define signals
    frame_changed = pyqtSignal(int, object)  # Current frame number, frame data
    video_loaded = pyqtSignal(str, int)  # Filename, total frames
    duplicate_frames_found = pyqtSignal(dict)  # Dictionary of duplicate frames
    
    def __init__(self, parent=None):
        """Initialize the VideoManager."""
        super().__init__(parent)
        
        # Initialize properties
        self.cap = None  # Video capture object
        self.video_filename = ""
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False
        self.playback_speed = 1.0
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)
        
        # Duplicate frame detection
        self.duplicate_frames_enabled = True
        self.duplicate_frames_cache = {}  # Maps frame hash to list of frame numbers
        self.frame_hashes = {}  # Maps frame number to its hash
        
        # Image dataset support
        self.is_image_dataset = False
        self.image_files = []
    
    def open_video(self, parent_window):
        """Open a video file dialog and load the selected video."""
        filename, _ = QFileDialog.getOpenFileName(
            parent_window,
            "Open Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )

        if filename:
            # Reset image dataset related state
            self.is_image_dataset = False
            self.image_files = []
            
            # Reset frame-related variables
            self.current_frame = 0
            self.frame_hashes = {}
            self.duplicate_frames_cache = {}
            
            return self.load_video_file(filename, parent_window)
        
        return False
    
    def load_video_file(self, filename, parent_window=None):
        """
        Load a video file and prepare it for playback.
        
        Args:
            filename (str): Path to the video file
            parent_window: Parent window for displaying error messages
            
        Returns:
            bool: True if video loaded successfully, False otherwise
        """
        # Close any existing video
        if self.cap:
            self.cap.release()

        # Open the video file
        self.cap = cv2.VideoCapture(filename)

        if not self.cap.isOpened():
            if parent_window:
                QMessageBox.critical(parent_window, "Error", "Could not open video file!")
            self.cap = None
            return False

        self.video_filename = filename
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0

        # Read the first frame
        ret, frame = self.cap.read()
        if ret:
            # Emit signals to notify about the loaded video
            self.video_loaded.emit(filename, self.total_frames)
            self.frame_changed.emit(0, frame)
            
            # Check if we need to scan for duplicate frames
            if parent_window and self.duplicate_frames_enabled and not self.frame_hashes:
                # Ask if user wants to scan for duplicates
                reply = QMessageBox.question(
                    parent_window,
                    "Duplicate Frame Detection",
                    "Would you like to scan this video for duplicate frames?\n"
                    "(This will help automatically propagate annotations)",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )

                if reply == QMessageBox.Yes:
                    QTimer.singleShot(500, lambda: self.scan_video_for_duplicates(parent_window))
            
            return True
        else:
            if parent_window:
                QMessageBox.critical(parent_window, "Error", "Could not read video frame!")
            self.cap.release()
            self.cap = None
            return False
    
    def get_current_frame(self):
        """Get the current frame data."""
        if not self.cap or not self.cap.isOpened():
            return None
            
        # Set the video position to the current frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        
        # Read the frame
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def goto_frame(self, frame_number):
        """
        Go to a specific frame in the video.
        
        Args:
            frame_number (int): The frame number to go to
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.cap or not self.cap.isOpened():
            return False
            
        # Ensure frame number is within valid range
        if frame_number < 0:
            frame_number = 0
        elif frame_number >= self.total_frames:
            frame_number = self.total_frames - 1
            
        # Set the current frame
        self.current_frame = frame_number
        
        # Get the frame data
        frame = self.get_current_frame()
        if frame is not None:
            # Emit signal with new frame
            self.frame_changed.emit(frame_number, frame)
            return True
        
        return False
    
    def next_frame(self):
        """Go to the next frame in the video."""
        if not self.cap or self.current_frame >= self.total_frames - 1:
            # Stop playback if we've reached the end
            if self.is_playing:
                self.toggle_play()
            return False
            
        return self.goto_frame(self.current_frame + 1)
    
    def prev_frame(self):
        """Go to the previous frame in the video."""
        if not self.cap or self.current_frame <= 0:
            return False
            
        return self.goto_frame(self.current_frame - 1)
    
    def toggle_play(self):
        """Toggle video playback."""
        if not self.cap:
            return
            
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            # Calculate interval based on playback speed and video FPS
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            interval = int(1000 / (fps * self.playback_speed))
            self.play_timer.start(max(1, interval))
        else:
            self.play_timer.stop()
    
    def set_playback_speed(self, speed):
        """
        Set the playback speed.
        
        Args:
            speed (float): Playback speed multiplier (1.0 = normal speed)
        """
        self.playback_speed = speed
        
        # Update timer interval if playing
        if self.is_playing and self.cap:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            interval = int(1000 / (fps * self.playback_speed))
            self.play_timer.setInterval(max(1, interval))
    
    def scan_video_for_duplicates(self, parent_window=None):
        """
        Scan the video for duplicate frames.
        
        This helps with automatically propagating annotations to identical frames.
        """
        if not self.cap or not self.cap.isOpened():
            return
            
        # Store current frame to restore later
        current_frame = self.current_frame
        
        # Import the hash function outside the loop
        try:
            from utils import calculate_frame_hash
        except ImportError:
            print("Warning: Could not import calculate_frame_hash from utils. Using a simple hash function instead.")
            # Define a simple hash function if import fails
            def calculate_frame_hash(img):
                # Simple hash: resize to small image and calculate average of pixels
                small_img = cv2.resize(img, (256, 256))
                gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
                avg = gray.mean()
                # Create a binary hash based on whether pixels are above or below average
                binary_hash = (gray > avg).flatten()
                # Convert binary array to string hash
                return ''.join(['1' if b else '0' for b in binary_hash])
        
        # Create progress dialog if parent window is provided
        progress_dialog = None
        if parent_window:
            from PyQt5.QtWidgets import QProgressDialog
            from PyQt5.QtCore import Qt
            
            # Create the progress dialog with proper parent and flags
            progress_dialog = QProgressDialog("Scanning for duplicate frames...", "Cancel", 0, self.total_frames, parent_window)
            progress_dialog.setWindowTitle("Duplicate Frame Detection")
            progress_dialog.setWindowModality(Qt.WindowModal)  # Make it modal to block parent window
            progress_dialog.setMinimumDuration(0)  # Show immediately
            progress_dialog.setValue(0)
            progress_dialog.setMinimumWidth(300)  # Ensure dialog is wide enough
            progress_dialog.setCancelButton(None)  # Optional: remove cancel button if you don't handle cancellation
            
            # Force the dialog to show and process events
            progress_dialog.show()
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()  # Process pending events to ensure dialog displays
        else:
            print("no parent window!!")
            print(parent_window)
        # Reset duplicate frame data
        self.frame_hashes = {}
        self.duplicate_frames_cache = {}
        
        try:
            # Scan all frames
            for frame_num in range(self.total_frames):
                # Check for cancellation
                if progress_dialog and progress_dialog.wasCanceled():
                    break
                    
                # Update progress
                if progress_dialog:
                    progress_dialog.setValue(frame_num)
                
                # Set position and read frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = self.cap.read()
                
                if ret:
                    try:
                        # Calculate frame hash
                        frame_hash = calculate_frame_hash(frame)
                        
                        # Store hash
                        self.frame_hashes[frame_num] = frame_hash
                        
                        # Add to duplicate cache
                        if frame_hash not in self.duplicate_frames_cache:
                            self.duplicate_frames_cache[frame_hash] = []
                        self.duplicate_frames_cache[frame_hash].append(frame_num)
                    except Exception as e:
                        print(f"Error processing frame {frame_num}: {str(e)}")
            
            # Clean up progress dialog
            if progress_dialog:
                progress_dialog.setValue(self.total_frames)
                progress_dialog.close()
            
            # Restore original frame
            self.goto_frame(current_frame)
            
            # Count duplicates
            duplicate_count = sum(
                len(frames) - 1
                for frames in self.duplicate_frames_cache.values()
                if len(frames) > 1
            )
            
            # Filter out non-duplicates from the cache
            self.duplicate_frames_cache = {
                hash_val: frames 
                for hash_val, frames in self.duplicate_frames_cache.items() 
                if len(frames) > 1
            }
            
            # Notify about results
            if parent_window:
                QMessageBox.information(
                    parent_window,
                    "Duplicate Frame Detection",
                    f"Found {duplicate_count} duplicate frames in the video.",
                )
            
            # Emit signal with duplicate frames
            self.duplicate_frames_found.emit(self.duplicate_frames_cache)
            
            return duplicate_count
            
        except Exception as e:
            # Handle any unexpected errors
            if progress_dialog:
                progress_dialog.close()
                
            if parent_window:
                QMessageBox.critical(
                    parent_window,
                    "Error",
                    f"An error occurred while scanning for duplicates: {str(e)}"
                )
            
            print(f"Error in scan_video_for_duplicates: {str(e)}")
            
            # Restore original frame
            self.goto_frame(current_frame)
            
            return 0
    
    def get_duplicate_frames(self, frame_number):
        """
        Get all frames that are duplicates of the given frame.
        
        Args:
            frame_number (int): The frame number to find duplicates for
            
        Returns:
            list: List of frame numbers that are duplicates of the given frame
        """
        if not self.duplicate_frames_enabled or frame_number not in self.frame_hashes:
            return []
            
        frame_hash = self.frame_hashes[frame_number]
        duplicates = self.duplicate_frames_cache.get(frame_hash, [])
        
        # Return all duplicates except the current frame
        return [f for f in duplicates if f != frame_number]
    
    def close_video(self):
        """Close the current video."""
        if self.cap:
            # Stop playback if active
            if self.is_playing:
                self.toggle_play()
                
            # Release the video capture
            self.cap.release()
            self.cap = None
            
            # Reset state
            self.video_filename = ""
            self.current_frame = 0
            self.total_frames = 0
            self.frame_hashes = {}
            self.duplicate_frames_cache = {}
            
            return True
        return False
    
    def check_for_annotation_files(self, video_filename, parent_window=None):
        """
        Check if annotation files with the same base name as the video exist.
        If found, ask the user if they want to import them.

        Args:
            video_filename (str): Path to the video file
            parent_window: Parent window for displaying dialogs
            
        Returns:
            list: List of found annotation files
        """
        if not parent_window:
            return []
            
        # Get the directory and base name without extension
        directory = os.path.dirname(video_filename)
        base_name = os.path.splitext(os.path.basename(video_filename))[0]

        # Check for auto-save file first
        autosave_file = os.path.join(directory, f"{base_name}_autosave.json")

        # List of possible annotation file extensions to check
        extensions = [".txt", ".json", ".xml"]

        # Find matching annotation files
        annotation_files = []
        for ext in extensions:
            potential_file = os.path.join(directory, base_name + ext)
            if os.path.exists(potential_file) and potential_file != autosave_file:
                annotation_files.append(potential_file)
                
        # Check if the json file in annotation_files is project save not a coco
        for an in list(annotation_files):  # Create a copy to safely modify during iteration
            if an.endswith(".json"):
                try:
                    with open(an, "r") as f:
                        data = json.load(f)
                        if "viat_project_identifier" in data:
                            annotation_files.remove(an)
                except:
                    pass  # If we can't read the file, keep it in the list
                    
        return annotation_files