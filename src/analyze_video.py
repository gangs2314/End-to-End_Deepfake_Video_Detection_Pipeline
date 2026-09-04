import subprocess
from predict_video import classify_video

VIDEO_PATH = "input/r3.mp4"

subprocess.run(
    ["python", "src/extract_frames.py", VIDEO_PATH],
    check=True
)

result, confidence = classify_video("temp_frames")

print("\nFINAL RESULT")
print("============")
print(f"Video Type : {result}")
print(f"Confidence : {confidence:.2f}")
