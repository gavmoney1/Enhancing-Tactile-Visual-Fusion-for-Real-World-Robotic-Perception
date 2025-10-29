import os
import cv2

# --- Config ---
input_root = r"input/dir/here"
target_size = (224, 224)  # width, height

# Walk through input directory
for root, dirs, files in os.walk(input_root):
    for fname in files:
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        fpath = os.path.join(root, fname)
        img = cv2.imread(fpath)
        if img is None:
            print(f"Could not open {fpath}")
            continue

        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(fpath, resized)
        print(f"Resized in place: {fpath}")

print("All images resized.")
