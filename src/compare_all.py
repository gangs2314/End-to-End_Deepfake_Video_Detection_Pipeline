import os
import subprocess
import sys  # <--- WE NEED THIS LIBRARY
from predict_video import classify_video

# List your 4 videos here
videos = ["input/r3.mp4", "input/r4.mp4", "input/random1.mp4", "input/random2.mp4"]

results = {}

print(f"{'VIDEO':<20} | {'PREDICTION':<15} | {'REALNESS SCORE':<15}")
print("-" * 55)

for video in videos:
    # 1. Extract frames
    # We change "python" to sys.executable to force it to use the VENV Python
    subprocess.run([sys.executable, "src/extract_frames.py", video], check=False, stdout=subprocess.DEVNULL)
    
    # 2. Predict
    try:
        label, score = classify_video("temp_frames")
    except Exception as e:
        label, score = "ERROR", 0.0
    
    # Store result
    results[video] = score
    print(f"{video:<20} | {label:<15} | {score:.4f}")

    # Clean up temp frames for next video
    if os.path.exists("temp_frames"):
        for f in os.listdir("temp_frames"):
            os.remove(os.path.join("temp_frames", f))

print("-" * 55)
if results:
    best_video = max(results, key=results.get)
    print(f"\n✅ The REAL video is likely: {best_video} (Highest Realness Score)")