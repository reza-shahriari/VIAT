import os
import glob
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg

def _rename_clips_sequentially(session_dir):
    """Rename clips in the session directory sequentially if they match the template."""
    clips = sorted(glob.glob(os.path.join(session_dir, "*-Scene-*.mp4")))
    for i, clip in enumerate(clips):
        new_name = os.path.join(session_dir, f"video_{i}.mp4")
        if os.path.exists(new_name):
            os.remove(new_name)
        os.rename(clip, new_name)

def split_video_by_scenes(video_path, session_dir, threshold=3.0):
    """
    Detect scenes in a video and split it into multiple clips.
    Original video is moved or deleted.
    """
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
        
    scene_list = detect(video_path, AdaptiveDetector(adaptive_threshold=threshold))
    num_videos = len(scene_list)
    print(f"Detected {num_videos - 1} cuts -> {num_videos} clips")

    for i, scene in enumerate(scene_list):
        print(f"  Scene {i+1}: Start {scene[0].get_timecode()} -> End {scene[1].get_timecode()}")

    if num_videos > 1:
        split_video_ffmpeg(
            video_path,
            scene_list,
            output_dir=session_dir,
            show_progress=True,
            output_file_template='$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4',
        )
        _rename_clips_sequentially(session_dir)
        
        # Original is removed ("cut" behavior)
        if os.path.exists(video_path):
            os.remove(video_path)
            
    else:
        # Move it
        new_path = os.path.join(session_dir, "video_0.mp4")
        if os.path.exists(new_path):
            os.remove(new_path)
        os.rename(video_path, new_path)

    print(f"Done: {session_dir}\n")
    return session_dir, num_videos
