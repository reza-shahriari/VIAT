import numpy as np
from ultralytics import SAM

# Create a dummy image
img = np.zeros((400, 400, 3), dtype=np.uint8)
img[100:300, 100:300, :] = 255  # White square

# Load a small SAM model (assuming it's already downloaded or it will download automatically)
try:
    model = SAM('sam2_s.pt') # sam2.1_s.pt or sam_b.pt
    print("Model loaded.")
except Exception as e:
    print("Failed to load model:", e)

print("\n--- Testing points=[[200, 200]] ---")
try:
    res = model(img, points=[[200, 200]], labels=[1], verbose=False)
    print("Result len:", len(res))
    if len(res) > 0:
        print("Boxes:", res[0].boxes)
        print("Masks:", res[0].masks)
except Exception as e:
    print("Error:", e)

print("\n--- Testing points=[200, 200] ---")
try:
    res = model(img, points=[200, 200], labels=[1], verbose=False)
    print("Result len:", len(res))
    if len(res) > 0:
        print("Boxes:", res[0].boxes)
        print("Masks:", res[0].masks)
except Exception as e:
    print("Error:", e)

print("\n--- Testing bboxes=[[100, 100, 300, 300]] ---")
try:
    res = model(img, bboxes=[[100, 100, 300, 300]], verbose=False)
    print("Result len:", len(res))
    if len(res) > 0:
        print("Boxes:", res[0].boxes)
        print("Masks:", res[0].masks)
except Exception as e:
    print("Error:", e)

print("\n--- Testing bboxes=[100, 100, 300, 300] ---")
try:
    res = model(img, bboxes=[100, 100, 300, 300], verbose=False)
    print("Result len:", len(res))
    if len(res) > 0:
        print("Boxes:", res[0].boxes)
        print("Masks:", res[0].masks)
except Exception as e:
    print("Error:", e)
