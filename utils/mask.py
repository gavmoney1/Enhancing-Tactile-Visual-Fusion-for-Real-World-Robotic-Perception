import os
import math
import cv2
import numpy as np
import shutil

# --- Config ---
input_root = r"/your/unmasked/images/dir/here"
output_root = r"/your/masked/images/output/dir/here"
img_size = 224
visible_frac = 0.10  # 10% visible in center

def mask_image_center(img, visible_frac=0.10, size=224):
    """Keep only center square, black out rest."""
    img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    H, W = size, size

    side = int(round(math.sqrt(visible_frac) * min(H, W)))
    cy, cx = H // 2, W // 2
    y0, x0 = cy - side // 2, cx - side // 2
    y1, x1 = y0 + side, x0 + side

    mask = np.zeros((H, W), dtype="uint8")
    mask[y0:y1, x0:x1] = 255

    visible_part = cv2.bitwise_and(img_resized, img_resized, mask=mask)
    return visible_part

# Create output folder if it doesn't exist
os.makedirs(output_root, exist_ok=True)

# Walk through input directory
for root, dirs, files in os.walk(input_root, topdown=False):
    for fname in files:
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        parent_folder = os.path.basename(root)
        new_name = f"{parent_folder}_{fname}"

        # --- Masked output ---
        in_path = os.path.join(root, fname)
        out_path = os.path.join(output_root, new_name)

        img = cv2.imread(in_path)
        if img is None:
            print(f"Could not open {in_path}")
            continue

        masked = mask_image_center(img, visible_frac, img_size)
        cv2.imwrite(out_path, masked)
        print(f"Saved masked image: {out_path}")

        # --- Move original to top-level input directory ---
        new_in_path = os.path.join(input_root, new_name)
        if in_path != new_in_path:
            shutil.move(in_path, new_in_path)
            print(f"Moved original image to: {new_in_path}")

    # Delete empty subfolders
    if root != input_root and not os.listdir(root):
        os.rmdir(root)
        print(f"Deleted empty folder: {root}")

print("Processing complete.")
