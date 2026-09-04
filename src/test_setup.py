import sys
import os

print(f"Python Executable: {sys.executable}")

try:
    import mediapipe as mp
    print(f"✅ MediaPipe imported successfully from: {mp.__file__}")
    
    mp_face = mp.solutions.face_detection
    print("✅ mp.solutions.face_detection is available.")
    
except ImportError:
    print("❌ ERROR: MediaPipe not installed.")
except AttributeError:
    print("❌ ERROR: 'solutions' attribute missing. DO YOU HAVE A FILE NAMED mediapipe.py IN THIS FOLDER?")
except Exception as e:
    print(f"❌ ERROR: {e}")

# Check if model exists
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/resnet18.pth"))
if os.path.exists(model_path):
    print(f"✅ Model found at: {model_path}")
else:
    print(f"❌ MODEL MISSING at: {model_path}")
    print("   -> You MUST run 'py src/train.py' before comparing videos.")